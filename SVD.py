import torch.nn as nn
import os
import torch

from Quantizer import Quantizer

bit = 4

class SVDLinear(nn.Module):
    def __init__(self, Linear, name, args):
        super().__init__()
        self.name = name
        if isinstance(Linear, nn.Linear):
            self.OriginLinear = Linear
        else:
            self.OriginLinear = Linear.OriginLinear
        self.W1_32 = None
        self.W2_32 = None
        self.W1_16 = None
        self.W2_16 = None
        self.HasBias = False
        self.RankRatio = args.rank_ratio

        if self.OriginLinear.bias is not None:
            self.HasBias = True
        else:
            self.HasBias = False

        if os.path.exists(weight_path + self.name + ".pt"):
            data = torch.load(weight_path + self.name + ".pt")
            self.RankRatio = data["RankRatio"]

    def forward(self,x):
        print(f"LAYER: {self.name}")
        print(f"COMPRESSION RATIO: {self.RankRatio}")
        print(f"SHAPE: {self.OriginLinear.weight.shape}")
        self.svd_linear_X(x)
        self.get_diff(x)
        return self.OriginLinear(x)

    def svd_linear_X(self, inputs):
        def SVD_decomposition_X(data, layer, rank_ratio):
            W = layer.weight.data.t()
            X = data
            if W.dtype == torch.bfloat16:
                W = W.float()
            if X.dtype == torch.bfloat16:
                X = X.float()

            r_k = max(1, int((1 - rank_ratio) * W.shape[0] * W.shape[1] / (W.shape[0] + W.shape[1])))
            def process_tensor(SX, n):
                squared = SX ** 2
                current_len = squared.shape[0]
                if current_len >= n:
                    result = squared[:n]
                else:
                    padding = torch.zeros(n - current_len, dtype=squared.dtype, device=squared.device)
                    result = torch.cat([squared, padding])
                return result

            XX = X.reshape(-1, X.shape[-1])
            UX, SX, VhX = torch.linalg.svd(XX, full_matrices=False)
            X_RANK = torch.count_nonzero(SX > 1e-4).item()
            SXX = process_tensor(SX, XX.shape[-1])

            SXX[SXX < 1e-4] = 1
            SS = VhX.transpose(0, 1) * SXX

            SS = SS.float()
            W = SS.transpose(0, 1) @ W
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            IMP0 = (S ** 2) * torch.sum(U[:X_RANK, :] ** 2, dim=0)
            L_Norm_u = 2 / 55.5
            L_Norm_v = 2 / 55.5
            IMP1 = (S * (L_Norm_u + L_Norm_v) * 1)  ** 2
            UUU = torch.inverse(SS.transpose(0,1))@U@torch.diag(S)
            QUA_U = convert_bf(UUU.t(), 4)
            QUA_U = QUA_U.t()
            QUA_V = convert_bf(Vh, 4)
            alpha = calculate_IMMP1_alpha(data, IMP0, IMP1, torch.inverse(SS.transpose(0,1)), U, S, Vh, QUA_U, QUA_V, r_k)
            rk_32 = alpha
            rk_16 = 4 * r_k - 3 * rk_32
            bool_mask = torch.zeros_like(IMP0, dtype=torch.bool)
            bool_mask[ :rk_16] = True
            U = U[:, bool_mask]
            Vh = Vh[bool_mask, :]
            S = S[bool_mask]
            del bool_mask
            torch.cuda.empty_cache()
            W1_final, W2_final = torch.inverse(SS.transpose(0,1))@U@torch.diag(S), Vh
            del UX, SX, VhX, XX, W, SS, S
            torch.cuda.empty_cache()
            return W1_final[:, :rk_32], W2_final[:rk_32, :], W1_final[:, rk_32:rk_16], W2_final[rk_32:rk_16, :]
        def convert_bf(data, acc):
            if acc == 32:
                if data.dtype != torch.float32:
                    return data.float()
            elif acc == 16:
                if data.dtype != torch.bfloat16:
                    return data.bfloat16()
            else:
                qua = Quantizer(shape=2)
                qua.configure(acc, perchannel=True, sym=False)
                qua.find_params(data, weight=True)
                data = qua.quantize(data)
                if data.dtype != torch.bfloat16:
                    return data.bfloat16()
            return data
        def calculate_IMMP1_alpha(data, IMP0, IMP1, SS, U, S, Vh, QUA_U, QUA_V, r_k=0):
            num_columns = QUA_U.shape[1]
            original = data @ (SS @ U @ torch.diag(S) @ Vh).bfloat16()
            turn = 20
            Sum = []
            for i in range(turn):
                sample_size = int(r_k * (1 - i/turn)) * 4
                if int(r_k * i / turn) + sample_size >= num_columns:
                    Sum.append(float('inf'))
                    continue
                L = list(range(int(r_k * i / turn), int(r_k * i / turn) + sample_size))

                U1 = (SS @ U @ torch.diag(S)).bfloat16()
                V1 = Vh.clone().bfloat16()
                U1[:, L] = QUA_U[:, L]
                V1[L, :] = QUA_V[L, :]
                U1 = U1[:, :L[-1]+1]
                V1 = V1[:L[-1]+1, :]
                sampled = U1 @ V1
                F1 = torch.norm((original.bfloat16() - data @ sampled), p='fro') ** 2
                F2 = IMP1[L].sum()
                if F2 == 0:
                    return float('inf')
                Sum.append( F1.item() )
            alpha1 = max(int(r_k * (Sum.index(min(Sum))-1) / turn), 0)
            alpha2 = min(int(r_k * (Sum.index(min(Sum))+1) / turn), r_k)
            RESULT = 9999999999999
            alpha = alpha1
            for i in range(alpha1, alpha2, 1):
                sample_size = 4 * (r_k - i)
                L = list(range(i, i + sample_size))
                if i + sample_size >= num_columns:
                    continue

                mask = torch.zeros(num_columns, dtype=torch.bool)
                mask[L] = True
                U1 = (SS @ U @ torch.diag(S)).bfloat16()
                V1 = Vh.clone().bfloat16()
                U1[:, L] = QUA_U[:, L]
                V1[L, :] = QUA_V[L, :]
                U1 = U1[:, :L[-1] + 1]
                V1 = V1[:L[-1] + 1, :]
                sampled = U1 @ V1
                F1 = torch.norm((original.bfloat16() - (data @ sampled).bfloat16()), p='fro') ** 2
                if F1 <= RESULT:
                    RESULT = F1
                    alpha = i
            return alpha

        W1weight32, W2weight32, W1weight16, W2weight16 = SVD_decomposition_X(inputs,self.OriginLinear,self.RankRatio)
        W1weight32 = convert_bf(W1weight32, 16)
        W2weight32 = convert_bf(W2weight32, 16)
        W1weight16 = convert_bf(W1weight16.t(), 4)
        W1weight16 = W1weight16.t()
        W2weight16 = convert_bf(W2weight16, 4)
        W1_32 = nn.Linear(W1weight32.shape[0], W1weight32.shape[1], bias=False).to(torch.bfloat16)
        W2_32 = nn.Linear(W2weight32.shape[0], W2weight32.shape[1], bias=self.HasBias).to(torch.bfloat16)
        W1_16 = nn.Linear(W1weight16.shape[0], W1weight16.shape[1], bias=False).to(torch.bfloat16)
        W2_16 = nn.Linear(W2weight16.shape[0], W2weight16.shape[1], bias=self.HasBias).to(torch.bfloat16)
        W1_32.weight.data = W1weight32.t()
        W2_32.weight.data = W2weight32.t()
        W1_16.weight.data = W1weight16.t()
        W2_16.weight.data = W2weight16.t()
        self.W1_32 = W1_32
        self.W2_32 = W2_32
        self.W1_16 = W1_16
        self.W2_16 = W2_16
        if self.HasBias:
            self.W2_32.bias = self.OriginLinear.bias
            print("Bias: ", self.OriginLinear.bias.shape)

        torch.save({'W1_32': self.W1_32, 'W2_32': self.W2_32, 'W1_16': self.W1_16, 'W2_16': self.W2_16, 'RankRatio': self.RankRatio},weight_path + self.name + ".pt")
        torch.cuda.empty_cache()

    def get_diff(self, inputs):
        x = inputs.to("cuda")
        print(x.dtype)
        a_i = torch.sum( (self.OriginLinear(x)-self.W2_32(self.W1_32(x))-self.W2_16(self.W1_16(x.bfloat16()))) ** 2 )
        print(f"LOSS PERSENT: {a_i/torch.sum((self.OriginLinear(x))**2)}, LOSS: {a_i}")
        self.W1_32 = None
        self.W2_32 = None
        self.W1_16 = None
        self.W2_16 = None
        torch.cuda.empty_cache()

class DIFFLayer(nn.Module):
    '''计算差分矩阵的'''
    def __init__(self, SVDLinear, name, args):
        super().__init__()
        self.W1_32 = None
        self.W2_32 = None
        self.W1_16 = None
        self.W2_16 = None
        self.Delta = None
        self.name = name
        self.OriginLinear = SVDLinear.OriginLinear
        self.SparsityRatio = args.sparsity_ratio
        self.RankRatio = SVDLinear.RankRatio

    def forward(self, x):
        if os.path.exists(weight_path + self.name + ".pt"):
            data = torch.load(weight_path + self.name + ".pt")
            self.W1_32 = data["W1_32"]
            self.W2_32 = data["W2_32"]
            self.W1_16 = data["W1_16"]
            self.W2_16 = data["W2_16"]
        print(f"LAYER: {self.name}")
        self.get_diff(x)
        torch.save({"W1_32": self.W1_32, "W2_32": self.W2_32,"W1_16": self.W1_16, "W2_16": self.W2_16, "RankRatio": self.RankRatio, "Delta": self.Delta},
                   weight_path + self.name + ".pt")
        self.W1_32 = None
        self.W2_32 = None
        self.W1_16 = None
        self.W2_16 = None
        self.Delta = None
        torch.cuda.empty_cache()
        return self.OriginLinear(x)

    def get_diff(self, X):
        def select_top_difference_index(E_i, sparsity_rate, delta_matrix):
            k = int(sparsity_rate * E_i.shape[0] * E_i.shape[1])
            _, flat_indices = torch.topk(E_i.abs().flatten(), k=k)

            mask = torch.ones_like(E_i, dtype=torch.bool).flatten()
            mask[flat_indices] = False
            mask = mask.reshape(E_i.shape)

            delta_matrix[mask] = 0
            return delta_matrix
        def save_difference_matrix(data, delta_matrix, sparsity_rate):
            X = data
            M = torch.zeros((delta_matrix.shape[0], delta_matrix.shape[1]))
            if X.dtype == torch.bfloat16:
                X = X.float()
            if delta_matrix.dtype == torch.bfloat16:
                delta_matrix = delta_matrix.float()
            if data.dim() == 3:
                XX = torch.zeros((X.shape[2], X.shape[2]))
                for x in X:
                    XX += (x.transpose(0, 1) @ x)
                M += (XX.sum(dim=0) @ delta_matrix * delta_matrix)
                return select_top_difference_index(M, sparsity_rate, delta_matrix)
            elif data.dim() == 2:
                M += (X.transpose(0, 1) @ X @ delta_matrix * delta_matrix)
                return select_top_difference_index(M, sparsity_rate, delta_matrix)
            else:
                print('save_difference_matrix Wrong')
        def convert_bf(data, acc):
            if acc == 32:
                if data.dtype != torch.float32:
                    return data.float()
            elif acc == 16:
                if data.dtype != torch.bfloat16:
                    return data.bfloat16()
            else:
                qua = Quantizer(shape=2)
                qua.configure(acc, perchannel=True, sym=False)
                qua.find_params(data, weight=True)  # 对权重张量x计算scale/zero
                data = qua.quantize(data)
                if data.dtype != torch.bfloat16:
                    return data.bfloat16()
            return data


        delta_matrix = self.OriginLinear.weight.data.t() - self.W1_32.weight.data.t() @ self.W2_32.weight.data.t() - self.W1_16.weight.data.t() @ self.W2_16.weight.data.t()
        DeltaWWeight = save_difference_matrix(X, delta_matrix, self.SparsityRatio)
        DeltaWWeight = convert_bf(DeltaWWeight, 4)
        DeltaW = nn.Linear(DeltaWWeight.shape[0], DeltaWWeight.shape[1], bias=False).to(torch.bfloat16)
        DeltaW.weight.data = DeltaWWeight.t()
        self.Delta = DeltaW

class SVDLayer(nn.Module):
    def __init__(self, Linear, name, args):
        super().__init__()
        self.W1_32 = Linear.W1_32
        self.W2_32 = Linear.W2_32
        self.W1_16 = Linear.W1_16
        self.W2_16 = Linear.W2_16
        self.DeltaW = Linear.Delta
        self.name = name

        if os.path.exists(weight_path + self.name + ".pt"):
            print(self.name)
            data = torch.load(weight_path + self.name + ".pt")
            self.W1_32 = data["W1_32"]
            self.W2_32 = data["W2_32"]
            self.W1_16 = data["W1_16"]
            self.W2_16 = data["W2_16"]
            self.DeltaW = data["Delta"]

            if self.W1_32 is None or self.W1_16 is None:
                print("W1 is None:")
            if self.W2_32 is None or self.W2_16 is None:
                print("W2 is None:")

        if args.not_delta:
            self.forward = self.forward_with_delta
        else:
            self.forward = self.forward_without_delta

    def forward_with_delta(self, x):
        return self.W2_32(self.W1_32(x)) + self.W2_16(self.W1_16(x.bfloat16()))+self.DeltaW(x.bfloat16())

    def forward_without_delta(self, x):
        return self.W2_32(self.W1_32(x)) + self.W2_16(self.W1_16(x.bfloat16()))

def factorize_model(model, original_layer_type, new_layer_type,name_detail="", args=None):
    original_class = globals().get(original_layer_type)
    if original_class is None:
        original_class = getattr(nn, original_layer_type, None)
    for name, module in model.named_children():
        if name == "lm_head":
            continue
        name_detail += (name + "_")
        if isinstance(module, original_class):
            new_class = globals().get(new_layer_type)
            if new_class is None:
                new_class = getattr(nn, new_layer_type, None)
            if new_class is None:
                raise ValueError(f"New layer type '{new_layer_type}' not found")
            setattr(model, name, new_class(module, name_detail, args))
        else:
            factorize_model(module, original_layer_type, new_layer_type, name_detail, args=args)
        name_detail = name_detail[: -len(name + "_")]
    return model

def get_max_tokens(tokenizer, sample_dataset):
    max_tokens = 0
    for text in sample_dataset:
        tokens = tokenizer.tokenize(text)
        current_length = len(tokens)
        if current_length > max_tokens:
            max_tokens = current_length
    print(f"the max tokens number: {max_tokens}")

def SVDModel(model, tokenizer, sample_dataset, args):
    global weight_path
    weight_path = args.data_save_path
    get_max_tokens(tokenizer, sample_dataset)
    model.eval()

    model = factorize_model(model,"Linear","SVDLinear",args=args).to("cuda")
    inputs = tokenizer(sample_dataset, return_tensors="pt", padding=True, truncation=True,max_length=args.max_length).to("cuda")
    print("="*20+"begin svd"+"="*20)
    try:
        with torch.no_grad():
            _ = model(**inputs)
    except Exception as e:
        raise e
    print("="*20+"end svd"+"="*20)
    model = factorize_model(model, "SVDLinear", "DIFFLayer", args=args).to("cuda")
    print("=" * 20 + "begin delta" + "=" * 20)
    try:
        with torch.no_grad():
            _ = model(**inputs)
    except Exception as e:
        raise e
    print("=" * 20 + "end delta" + "=" * 20)
    model = factorize_model(model, "DIFFLayer", "SVDLayer", args=args).to("cuda")
    print("=" * 20 + "end total" + "=" * 20)

    return model
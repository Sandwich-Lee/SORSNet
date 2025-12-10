import torch

def cov_matrix(input_m, input_cls_m = None):
    """
    input_m: b,c,t*h*w
    input_cls: b,c
    """
    eps = 0.01
    mean = torch.mean(input_m, 2, keepdim=True) # b, c, 1
    x = input_m - mean.expand(-1, -1, input_m.size(2)) #
    output = torch.bmm(x, x.transpose(1, 2))
    output = output / (input_m.size(1)-1)
    # output = output + eps * torch.eye(output.size(1), device=output.device).unsqueeze(0)
    # print(output.shape)

    # # -------------------

    if input_cls_m is not None:
        input_cls_m = input_cls_m.unsqueeze(2)  # b,c,1
        # last column
        cov_with_col = torch.cat([output, input_cls_m], dim=2)
        # last raw
        input_cls_m_T = input_cls_m.transpose(1, 2)  # (b, 1, c)

        bottom_row = torch.cat([input_cls_m_T, torch.ones(mean.size(0), 1, 1, device=mean.device, dtype=mean.dtype)], dim=2)
        output = torch.cat([cov_with_col, bottom_row], dim=1)  # (b, c+1, c+1)
    return output

if __name__ == '__main__':
    m = torch.tensor(torch.randn(2, 4, 2))
    cov_matrix(m)
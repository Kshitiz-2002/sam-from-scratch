import torch 
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, Type, Optional

from .common import LayerNorm2d, MLPBlock

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,   
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 1024, 1024)
        x = self.patch_embed(x)
        # (B, 3, 1024, 1024) -> (B, 64, 64, 768)     conv stride-16, then permute to channels-last

        if self.pos_embed is not None:
            x = x + self.pos_embed
            # (B, 64, 64, 768) + (1, 64, 64, 768) -> (B, 64, 64, 768)   broadcasts over batch

        for block in self.blocks:
            x = block(x)
            # (B, 64, 64, 768) -> (B, 64, 64, 768), repeated 12 times
            # 8 of these run windowed (14x14 windows internally), 4 run global (full 64x64)

        x = self.neck(x.permute(0, 3, 1, 2))
        # permute: (B, 64, 64, 768) -> (B, 768, 64, 64)     channels-last -> channels-first
    # neck:    (B, 768, 64, 64) -> (B, 256, 64, 64)     1x1 conv (768->256) + 3x3 conv (256->256), each + LayerNorm2d

        return x # (B, 256, 64, 64) 

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)   e.g. global block (1, 64, 64, 768)  
        shortcut = x
        x = self.norm1(x)

        # Windown partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            # (B, H, W, C) -> (B*num_windows, ws, ws, C)   e.g. (1,64,64,768) -> (25,14,14,768)

        x = self.attn(x)
        # windowed: (B*num_windows, ws, ws, C) -> (B*num_windows, ws, ws, C)   e.g. (25,14,14,768)
        # global:   (B, H, W, C) -> (B, H, W, C)                                e.g. (1,64,64,768)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
            # (B*num_windows, ws, ws, C) -> (B, H, W, C)   e.g. (25,14,14,768) -> (1,64,64,768)

        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        # norm2: (B,H,W,C)->(B,H,W,C)
        # mlp:   (B,H,W,C)->(B,H,W,4C)->(B,H,W,C)     (mlp_ratio=4.0 expands then contracts)
        # residual 2: (B, H, W, C)
        return x # (B, H, W, C) 

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(
        self, 
        in_chans: int = 3, 
        embed_dim: int = 768, 
        kernel_size: Tuple[int, int] = (16, 16), 
        stride: int = 16,
        padding: Tuple[int, int] = (0,0),    
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/16, W/16)
        x = self.proj(x)
        # (B, emed_dim, H/16, W/16) -> (B, H/16, W/16, embed_dim)
        x = x.permute(0, 2, 3, 1)
        return x
    
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = True, 
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]]=None
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
    
        # (B, H/16, W/16, embed_dim) -> (B, H/16, W/16, 3 * embed_dim)
        self.qkv = nn.Linear(dim, dim*3,bias=qkv_bias)
        # (B, H/16, W/16, embed_dim) -> (B, H/16, W/16, embed_dim)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert(
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # (B, H/16, W/16, embed_dim) -> (B, H/16, W/16, 3 * embed_dim)
        qkv = self.qkv(x)
        # (B, H/16, W/16, 3 * embed_dim) -> (B, H/16*W/16, 3, num_heads, embed_dim/num_heads)
        qkv = qkv.reshape(B, H*W, 3, self.num_heads, -1)
        # (B, H/16*W/16, 3, num_heads, embed_dim/num_heads) -> (3, B, num_heads, H/16*W/16, embed_dim/num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (B, H/16*W/16, 3, num_heads, embed_dim/num_heads) -> (B, num_heads, H/16*W/16, embed_dim/num_heads) each
        q, k, v = qkv.unbind(0)
        
        # (1, B, num_heads, H/16*W/16, embed_dim/num_heads) -> (B*num_heads, H/16*W/16, embed_dim/num_heads)
        q = q.reshape(B*self.num_heads, H*W, self.head_dim)
        # (1, B, num_heads, H/16*W/16, embed_dim/num_heads) -> (B*num_heads, H/16*W/16, embed_dim/num_heads)
        k = k.reshape(B*self.num_heads, H*W, self.head_dim)
        # (1, B, num_heads, H/16*W/16, embed_dim/num_heads) -> (B*num_heads, H/16*W/16, embed_dim/num_heads)
        v = v.reshape(B*self.num_heads, H*W, self.head_dim)

        # (B*num_heads, H/16*W/16, embed_dim/num_heads) * (B*num_heads, embed_dim/num_heads, H/16*W/16) -> (B*num_heads, H/16*W/16, H/16*W/16)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        # (B*num_heads, H/16*W/16, H/16*W/16) -> (B*num_heads, H/16*W/16, H/16*W/16)
        attn = torch.softmax(attn, dim=-1)
        x = attn @ v
        # (B*num_heads, H/16*W/16, embed_dim/num_heads) -> (B, num_heads, H/16, W/16, embed_dim/num_heads) -> (B, H/16, W/16, num_heads, embed_dim/num_heads)
        # (B, H/16, W/16, num_heads, embed_dim/num_heads) -> (B, H/16, W/16, embed_dim)
        x = x.reshape(B, self.num_heads, H, W, self.head_dim).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1) 
        # (B, H/16, W/16, embed_dim) -> (B, H/16, W/16, embed_dim)
        x = self.proj(x)
        return x
    

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size: size of the query q
        k_size: size of the key k
        rel_pos(Tensor): the learnable table, relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """

    # 1. The table must have 2*max(q_size, k_size) - 1 rows. If not, interpolate.
    max_rel_dist = 2 * max(q_size, k_size) - 1
    if rel_pos.shape[0] != max_rel_dist:
        # interpolate rel_pos to the right number of rows
        # rel_pos.shape[0] = L
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), # (rel_pos.shape[0], head_dim) -> (1, L, head_dim) -> (1, head_dim, L)
            size=max_rel_dist,
            mode="linear",
        ) # (1, head_dim, L) -> (1, head_dim, max_rel_dist)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0) # (1, head_dim, max_rel_dist) -> (head_dim, max_rel_dist) -> (max_rel_dist, head_dim)
    else:
        # (max_rel_dist, head_dim)
        rel_pos_resized = rel_pos

    # 2. Build query and key coordinate vectors, scaled if sizes differ.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0) # (q_size, 1)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0) # (1, k_size)

    # 3. Relative coords -> table indices (shift to be non-negative).
    # (q_size, 1) - (1, k_size) -> gets broadcasted -> (q_size, k_size)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    # 4. Gather the rows.
    return rel_pos_resized[relative_coords.long()] # (q_size, k_size, head_dim)


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int] 
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings 
    Args:
        attn (Tensor): attention map. (B*num_heads, q_h*q_w, k_h*k_w)
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C or head_dim).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    
    Rh = get_rel_pos(q_h, k_h, rel_pos_h) # (q_h, k_h, head_dim)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w) # (q_w, k_w, head_dim)

    B, _, h_dim = q.shape 
    r_q = q.reshape(B, q_h, q_w, h_dim) # (B, q_h*q_w, head_dim) -> (B, q_h, q_w, head_dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)   # (B*num_heads, q_h, q_w, head_dim), (q_h,k_h,head_dim) -> (B*num_heads, q_h, q_w, k_h)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)   # (B*num_heads, q_h, q_w, head_dim), (q_w,k_w,head_dim) -> (B*num_heads, q_h, q_w, k_w)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) # (B*num_heads, q_h*q_w, k_h*k_w) -> (B*num_heads, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None] # (B*num_heads,q_h,q_w,k_h) -> (B*num_heads,q_h,q_w,k_h,1), broadcasts over k_w
        + rel_w[:, :, :, None, :] # (B*num_heads,q_h,q_w,k_w) -> (B*num_heads,q_h,q_w,1,k_w), broadcasts over k_h
    ).view(B, q_h * q_w, k_h * k_w) # back to (B*num_heads, q_h*q_w, k_h*k_w)


    return attn


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size # e.g. (14 - 64%14) % 14 = (14-8)%14 = 6
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        # (B, H, W, C) -> (B, H+pad_h, W+pad_w, C)   e.g. (1, 64, 64, 768) -> (1, 70, 70, 768)

    Hp, Wp = H + pad_h, W + pad_w  # 70, 70
    
    x = x.view(B, Hp//window_size, window_size, Wp//window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # (B, Hp, Wp, C) -> (B, Hp/ws, ws, Wp/ws, ws, C)   e.g. (1,70,70,768) -> (1, 5, 14, 5, 14, 768)
    # permute: (B, Hp/ws, ws, Wp/ws, ws, C) -> (B, Hp/ws, Wp/ws, ws, ws, C)   e.g. (1,5,5,14,14,768)
    # view:    (B, Hp/ws, Wp/ws, ws, ws, C) -> (B*Hp/ws*Wp/ws, ws, ws, C)     e.g. (25, 14, 14, 768)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    # windows: (B*num_windows, ws, ws, C)   e.g. (25, 14, 14, 768)
    Hp, Wp = pad_hw # 70, 70
    H, W = hw  # 64, 64
    # 25 // (70*70 // 14 // 14) = 25 // 25 = 1
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    # (B*num_windows, ws, ws, C) -> (B, Hp/ws, Wp/ws, ws, ws, C)   e.g. (1, 5, 5, 14, 14, 768)

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    # permute: (B, Hp/ws, Wp/ws, ws, ws, C) -> (B, Hp/ws, ws, Wp/ws, ws, C)   e.g. (1,5,14,5,14,768)
    # view:    (B, Hp/ws, ws, Wp/ws, ws, C) -> (B, Hp, Wp, C)                 e.g. (1, 70, 70, 768)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
        # crop back to original size: (B, Hp, Wp, C) -> (B, H, W, C)   e.g. (1, 70, 70, 768) -> (1, 64, 64, 768)

    return x

import os 
import sys 
from difflib import SequenceMatcher
from tqdm import tqdm
import logging
from typing import Optional, Union, Tuple, List, Dict, Callable
import torch
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    CLIPFeatureExtractor,
    CLIPTokenizer,
    CLIPTextModel,
    UNet2DConditionModel,
    AutoencoderKL,
    PNDMScheduler
)
# from transformers import CLIPVisionModel
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np
import cv2
from PIL import Image
from einops import rearrange
import math

curdir = os.path.dirname(__file__)
rtpath = curdir
sys.path.append(rtpath)
logging.basicConfig(level=logging.INFO)
base_logger = logging.getLogger("p2p")

# ---------------- Global parameter ----------------
MODEL_PATH = "CompVis/stable-diffusion-v1-4"  
LOW_RESOURCE = True
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
DEFAULT_IMG_HEIGHT = 512
DEFAULT_IMG_WIDTH = 512
MAX_VIS_PIXEL_NUM = 32 * 32
BLEND_PIXEL_NUM = 16 * 16
BLEND_THRED = 0.3
SEED = 8888
TORCH_DTYPE = torch.float32
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH, 
    torch_dtype=TORCH_DTYPE,
    ).to(device)
# clip_img_encoder = CLIPVisionModel.from_pretrained(os.path.join(MODEL_PATH, "image_encoder"))

tokenizer = ldm_stable.tokenizer
cached_rtpath = os.path.join(curdir, "cached")
# ----------------------------------------------------


def extend_list(*args) -> List:
    out_list = []
    for i in args:
        if isinstance(i, str):
            out_list.append(i)
        elif isinstance(i, List):
            out_list.extend(i)
        elif isinstance(i, Tuple):
            out_list.extend(list(i))
        else:
            raise TypeError(f"*args type must be in [str, list, tuple], but received {type(i)}")
    return out_list


def checkdir(path: Union[str, List[str]], logger=base_logger) -> None:
    """check whether path exist, if not, create it! """
    def single_dir_check(path):
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"path have not exist, create it!")
        else:
            logger.info(f"path have exist!")
    if isinstance(path, str):
        single_dir_check(path)
        return
    elif isinstance(path, (list, tuple)):
        _ = [single_dir_check(i) for i in path]
    else:
        raise TypeError


@dataclass
class CachedAttnInfo:
    place_in_unet: str
    attn: Optional[torch.Tensor]
    is_cross: bool
    text_branch_flag: bool  # noise branch or text branch. if False, noise branch, else text branch
    token_attn_map: Optional[Dict[str, np.ndarray]] = None

    def __post_init__(self):
        self.place_in_block = self.place_in_unet.split(".")[0].split("_")[0]  # down, up, mid 
    
    def get_token_attn_map(self, token_ls) -> Dict[str, np.ndarray]:
        """
        Args:
            token_ls:
        """
        assert self.attn.ndim == 3, f"attn ndim must be equal to 3, but received: {self.attn.ndim}"
        score = attention_map_to_image(self.attn[..., :len(token_ls)]).cpu().numpy()
        token_attn_map = OrderedDict()
        for idx, token in enumerate(token_ls):
            token_attn_map[token] = score[..., idx]
        return token_attn_map


@dataclass
class Text2ImageResult:
    image: Optional[np.ndarray] = None
    latent: Optional[torch.FloatTensor] = None
    cached_attn_with_step: Optional[List[Dict[str, Dict[str, CachedAttnInfo]]]] = None
    cached_latent_with_step: Optional[List[torch.Tensor]] = None
    

@dataclass
class ImageEditResult:
    target_image: Optional[np.ndarray] = None
    source_image: Optional[np.ndarray] = None
    latent: Optional[torch.Tensor] = None
    cached_attn_with_step: Optional[List[Dict[str, Dict[str, CachedAttnInfo]]]] = None
    cached_latent_with_step: Optional[List[torch.Tensor]] = None
    cached_source_result: Optional[Text2ImageResult] = None


@dataclass
class EditParams:
    source_prompt: str
    target_prompt: str
    cross_merge_end_step: Union[float, int]  # cross attention merge step, 0-(cross_merge_step * diffusion step)  using cross-attn injection 
    self_merge_end_step: Union[float, int]  # self attention merge step, 0-(self_merge_step * diffusion step) using self-attn injection
    cross_merge_start_step: Union[float, int] = 0  # cross attention merge step, 0-(cross_merge_step * diffusion step)  using cross-attn injection
    self_merge_start_step: Union[float, int] = 0  # self attention merge step, 0-(self_merge_step * diffusion step) using self-attn injection
    addition_token_control_info: Optional[Dict] = None
    do_noise_branch_control: bool = False
    do_local_blend: bool = False  # using local blend
    blend_focus_text: Optional[List] = None

    def __post_init__(self):
        self.tokenizer = tokenizer
        self.device = device

        self.cross_merge_start_step = self.to_int(self.cross_merge_start_step)
        self.cross_merge_end_step = self.to_int(self.cross_merge_end_step)
        self.self_merge_start_step = self.to_int(self.self_merge_start_step)
        self.self_merge_end_step = self.to_int(self.self_merge_end_step)
        if self.addition_token_control_info is not None:
            self.addition_token_control_info = {k: float(v) for k, v in self.addition_token_control_info.items()}

        self.replace_index_map = self.get_replace_index_map(
            self.source_prompt,
            self.target_prompt,
            cross_merge_end_step=self.cross_merge_end_step,
            addition_token_control_info=self.addition_token_control_info
        )
        if self.do_local_blend:
            self.blend_token_map = self.get_blend_token_map(self.blend_focus_text)
            self.local_blend_obj = LocalBlend()
        else:
            self.local_blend_obj: Callable = lambda x, *_: x

    @staticmethod
    def to_int(step: Union[float, int]):
        if isinstance(step, float):
            return int(NUM_DIFFUSION_STEPS * step)
        return int(step)
    
    def set_params(self, obj):
        for k, v in self.__dict__.items():
            if hasattr(obj, k):
                setattr(obj, k, v)

    def get_replace_index_map(
            self,
            source_prompt: str,
            target_prompt: str,
            cross_merge_end_step: int,
            addition_token_control_info: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            source_prompt:
            target_prompt:
            cross_merge_end_step
            addition_token_control_info
        """

        def get_token_weight(_token_ls, _token_weight_map, max_token_num, default_value=1.0):
            token_weight = torch.ones(max_token_num)
            for idx, tok in enumerate(_token_ls):
                token_weight[idx] = _token_weight_map.get(tok, default_value)
            return token_weight

        source_prompt_token_ls = [self.tokenizer.decode(i) for i in self.tokenizer.encode(source_prompt)]
        target_prompt_token_ls = [self.tokenizer.decode(i) for i in self.tokenizer.encode(target_prompt)]
        source_token_len: int = len(source_prompt_token_ls)
        target_token_len: int = len(target_prompt_token_ls)
        max_len: int = max(source_token_len, target_token_len)
        token_weight_map = dict()
        if addition_token_control_info is not None:
            for k, weight in addition_token_control_info.items():
                cur_token_id_ls = self.tokenizer.encode(k)[1:-1]
                for cur_token_id in cur_token_id_ls:
                    token_weight_map[self.tokenizer.decode(cur_token_id)] = weight

        replace_index_map = {}
        assert max_len <= MAX_NUM_WORDS

        source_mask = torch.Tensor([False] * MAX_NUM_WORDS).bool().to(self.device)
        target_mask = torch.Tensor([False] * MAX_NUM_WORDS).bool().to(self.device)
        source_token_weight = get_token_weight(
            source_prompt_token_ls, token_weight_map, MAX_NUM_WORDS).to(device=self.device, dtype=TORCH_DTYPE)
        target_token_weight = get_token_weight(
            target_prompt_token_ls, token_weight_map, MAX_NUM_WORDS).to(device=self.device, dtype=TORCH_DTYPE)
        target_mask[max_len:] = True
        source_mask[max_len:] = True

        matcher = SequenceMatcher(None, source_prompt_token_ls, target_prompt_token_ls)
        for i in matcher.get_opcodes():
            tag, i1, i2, j1, j2 = i
            if tag == "equal":
                source_mask[i1: i2] = True
                target_mask[j1: j2] = True
            elif tag == "replace":
                if i2 - i1 == j2 - j1:
                    source_mask[i1: i2] = True
                    target_mask[j1: j2] = True
                elif i2 - i1 > j2 - j1:
                    source_mask[i1: i1 + j2 - j1] = True
                    target_mask[j1: j2] = True
                else:
                    source_mask[i1: i2] = True
                    target_mask[j1: j1 + i2 - i1] = True
            else:
                pass
        replace_index_map["source_mask"] = source_mask
        replace_index_map["target_mask"] = target_mask
        replace_index_map["source_token_weight"] = rearrange(source_token_weight, "c -> 1 1 1 c")
        replace_index_map["target_token_weight"] = rearrange(target_token_weight, "c -> 1 1 1 c")
        return replace_index_map

    def get_blend_token_map(self, blend_focus_text: List[str]) -> torch.Tensor:
        prompt_ls = [self.source_prompt, self.target_prompt]
        assert blend_focus_text is not None
        assert blend_focus_text.__len__() == 2, \
            f"get blend focus text len: {len(blend_focus_text)},\n{blend_focus_text}"
        blend_token_map = torch.zeros((2, 1, 1, 1, 1, MAX_NUM_WORDS))  # (b, h, c, h, w, d)
        for idx, text in enumerate(blend_focus_text):
            cur_match_id_ls = self.get_match_id(prompt_ls[idx], text)
            # cur_token_id_ls: List = self.tokenizer.encode(text)[1:-1]
            blend_token_map[idx, ..., cur_match_id_ls] = 1
        return blend_token_map.to(device=self.device, dtype=TORCH_DTYPE)

    def get_match_id(self, prompt: str, word: str):
        prompt_id_ls = self.tokenizer.encode(prompt)[1:-1]
        word_id_ls = self.tokenizer.encode(word)[1:-1]
        match_id_ls = [prompt_id_ls.index(i) + 1 for i in word_id_ls]
        return match_id_ls


def attention_map_to_image(attn: torch.Tensor) -> torch.Tensor:
    """
    Args:
        attn: (Head, pixel_num, dim)
    Return:
        reduce_head: (h, w, dim)
    """
    h, p, d = attn.shape[:3]
    width = int(math.sqrt(p)) 
    seq2img = rearrange(attn, "H (h w) d -> H h w d", w=width)
    reduce_head = seq2img.sum(0, keepdim=False) / seq2img.shape[0]
    return reduce_head


def draw_heat_map(score: np.ndarray, aim_shape: Optional[tuple] = None):
    """
    aim_shape: tuple of reshape size (w, h)
    """
    assert score.ndim == 2, f"score ndim must be equal to 2, but received: {score.ndim}"
    score_norm = (score - np.min(score)) / (np.max(score) - np.min(score))
    score_norm = np.clip(score_norm, 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * score_norm), cv2.COLORMAP_JET)
    if aim_shape is not None:
        heatmap = cv2.resize(heatmap, aim_shape)
    return heatmap[..., ::-1].astype(np.uint8)


def draw_cross_attention_per_layer_all_time_step(
        token_ls: List[str], 
        cached_attn_with_step: Optional[List[Dict[str, Dict[str, CachedAttnInfo]]]],
        aim_shape: Tuple[int, int],
        text_branch_flag: bool = True,
        ) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """
    Args:
        token_ls: List[str]
        cached_attn_with_step: Optional[List[Dict[str, Dict[str, CachedAttnInfo]]]],
        aim_shape: tuple of reshape size (w, h)
        text_branch_flag: True
    Return:
        heat_map_info: Dict[str, List[Dict[str, np.ndarray]]]
    """
    def token_attn_map_to_heatmap(token_attn_map: Dict, cur_aim_shape):
        token_heatmap = OrderedDict()
        for k, v in token_attn_map.items():
            token_heatmap[k] = draw_heat_map(v, aim_shape=cur_aim_shape)
        return token_heatmap
    
    assert cached_attn_with_step is not None, "cached_attn_with_step_and_heat_map is None"
    heat_map_info = dict()
    for cur_step_cached_attn_info in cached_attn_with_step:
        merge_info = {}
        for cur_place, cur_attn_info in cur_step_cached_attn_info.items():
            cur_attn_info = cur_attn_info[f"text_control_flag_{text_branch_flag}"]
            if not cur_attn_info.is_cross:
                continue
            cur_attn_token_map = cur_attn_info.get_token_attn_map(token_ls=token_ls)
            heat_map_info.setdefault(cur_place, []).append(token_attn_map_to_heatmap(cur_attn_token_map, aim_shape))
            merge_info.setdefault(cur_attn_info.place_in_block, []).append(cur_attn_token_map)
        merge_all = []
        for cur_block, cur_block_attn_map in merge_info.items():
            new_token_attn_map = OrderedDict()
            for token in token_ls:
                cur_token_attn_map = np.mean(
                    np.stack([cv2.resize(i[token], aim_shape) for i in cur_block_attn_map], axis=0),
                    axis=0)
                new_token_attn_map[token] = cur_token_attn_map
            merge_all.append(new_token_attn_map)
            heat_map_info.setdefault(cur_block, []).append(token_attn_map_to_heatmap(new_token_attn_map, aim_shape))

        new_token_attn_map = OrderedDict()
        for token in token_ls:
            cur_token_heatmap = np.mean(np.stack([i[token] for i in merge_all], axis=0), axis=0)
            new_token_attn_map[token] = cur_token_heatmap
        heat_map_info.setdefault("merge_all", []).append(token_attn_map_to_heatmap(new_token_attn_map, aim_shape))

    return heat_map_info


class LocalBlend:

    def __init__(self, target_pixel_token: int = BLEND_PIXEL_NUM, k: int = 1, thred: float = BLEND_THRED):
        self.target_pixel_token = target_pixel_token
        self.k = k
        self.thred = thred

    def __call__(
            self,
            x_t: torch.Tensor,
            blend_token_mask: torch.Tensor,
            attn_map: Dict[str, Dict[str, CachedAttnInfo]]
    ) -> torch.Tensor:
        assert x_t.shape[0] == blend_token_mask.shape[0], \
            f"x_t shape: {x_t.shape}, blend_token_mask shape: {blend_token_mask.shape}"
        batch_size = x_t.shape[0]
        valid_map = self.get_aim_map(batch_size, attn_map)
        mask = (valid_map * blend_token_mask).sum(dim=-1).mean(dim=1)  # (batch_size, 1, 16, 16)
        mask = F.max_pool2d(mask, (self.k * 2 + 1, self.k * 2 + 1), (1, 1), padding=(self.k, self.k))  # 2, 1, 16, 16, 相当于做一个膨胀（取大的值）
        mask = F.interpolate(mask, size=(x_t.shape[2:]))  # 2, 1, 64, 64
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]  # normalize
        mask = mask.gt(self.thred)
        mask = (mask[:1] + mask[1:]).float()  # 1 1 64, 64
        ref_x_t = x_t[:1]
        x_t = ref_x_t + mask * (x_t - ref_x_t)
        return x_t

    def get_aim_map(self, batch_size: int, attn_map: Dict[str, Dict[str, CachedAttnInfo]]) -> torch.Tensor:
        valid_map = [vv.attn for k, v in attn_map.items() for kk, vv in v.items()
                     if vv.is_cross and vv.attn.shape[-2] == self.target_pixel_token and vv.text_branch_flag]
        maps = [item.reshape(batch_size, -1, 1, 16, 16, MAX_NUM_WORDS) for item in valid_map]  # b * head * 1 * h * w * dim
        maps = torch.cat(maps, dim=1)  # batch_size, 40, 1, 16, 16, 77
        return maps


class BaseController:
    def __init__(
            self,
            max_vis_pixel_num=MAX_VIS_PIXEL_NUM,
            cached_attn_info_flag: bool = False,
            edit_params: Optional[EditParams] = None,
            logger=base_logger
    ):
        """

        Args:
            max_vis_pixel_num: max cached attn
            cached_attn_info_flag:
            edit_params:
            logger:
        """
        self.num_att_layers = 0
        self.cross_attn_name_ls = None
        self.cached_attn_info_flag = cached_attn_info_flag
        self.max_vis_pixel_num = max_vis_pixel_num
        self.logger = logger
        self.text_branch_flag = None
        # True: diffusion with text input, else diffusion without null text input,
        # see the principle of classifier-free text-to-image generation
        self.current_step = None
        self.batch_size = None
        self.edit_params = edit_params

        # control params 
        self.replace_index_map = None
        self.cross_merge_start_step = 0
        self.cross_merge_end_step = NUM_DIFFUSION_STEPS
        self.self_merge_start_step = 0
        self.self_merge_end_step = NUM_DIFFUSION_STEPS
        self.do_noise_branch_control = False
        self.do_local_blend = False
        self.local_blend_obj = lambda x, *_: x

        self.init_edit_params(edit_params)

        # cached info
        self.not_control_attn_name_set = set()
        self.source_attn_map = {}
        self.blend_attn_map = {}
        self.cached_attn = {}
        self.cached_attn_name_set = set()

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
    
    def set_step(self, step: int):
        self.current_step = step

    def set_text_branch_flag(self, text_branch_flag: bool):
        self.text_branch_flag = text_branch_flag
    
    def set_step_source_attn_map(self, source_attn_map: Dict[str, CachedAttnInfo]):
        self.source_attn_map = source_attn_map

    def init_edit_params(self, edit_params: Optional[EditParams]):
        if edit_params is None:
            return 
        edit_params.set_params(self)
        self.logger.debug(f"{repr(edit_params)}")
    
    def reset_cached_info(self):
        self.not_control_attn_name_set = set()
        self.source_attn_map = {}
        self.blend_attn_map = {}
        self.cached_attn = {}
        self.cached_attn_name_set = set()

    @property
    def control_flag_all_step(self) -> bool:
        if self.do_noise_branch_control or (not self.do_noise_branch_control and self.text_branch_flag):
            return True
        else:
            return False

    @property
    def is_cross_attn_control_step(self):
        return self.cross_merge_start_step <= self.current_step <= self.cross_merge_end_step

    @property
    def is_self_attn_control_step(self):
        return self.self_merge_start_step <= self.current_step <= self.self_merge_end_step

    @property
    def do_cross_attn_control_flag(self) -> bool:
        return self.control_flag_all_step and self.is_cross_attn_control_step

    @property
    def do_self_attn_control_flag(self) -> bool:
        return self.control_flag_all_step and self.is_self_attn_control_step

    def set_cached_attn(self, place_in_unet, is_cross, attn):
        self.cached_attn[place_in_unet][
            f"text_control_flag_{self.text_branch_flag}"] = CachedAttnInfo(
            place_in_unet, attn, is_cross, self.text_branch_flag)

    def set_blend_attn_map(self, place_in_unet, is_cross, attn):
        if self.blend_attn_map.get(place_in_unet, None) is None:
            self.blend_attn_map[place_in_unet] = dict()
        self.blend_attn_map[place_in_unet][
            f"text_control_flag_{self.text_branch_flag}"] = CachedAttnInfo(
            place_in_unet, attn, is_cross, self.text_branch_flag)

    def get_source_attn(self, place_in_unet) -> torch.Tensor:
        assert isinstance(self.source_attn_map, dict), \
            f"you should set source_attn_map by `self.set_step_source_attn_map`!"
        return self.source_attn_map[place_in_unet][f"text_control_flag_{self.text_branch_flag}"].attn

    def control_info_checking(self):
        assert isinstance(self.edit_params, EditParams), \
            f"edit_params must be EditParams, but received: {type(self.edit_params)}"

    def control(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        # print(f">>>cached_attn_flag: {self.cached_attn_info_flag}")
        assert self.current_step is not None, f"please set current time step by 'self.set_step'!"
        pixel_num = attn.shape[1]
        if pixel_num > self.max_vis_pixel_num:
            self.not_control_attn_name_set.add(place_in_unet)
            return attn
        if place_in_unet not in self.cached_attn.keys():
            self.cached_attn[place_in_unet] = dict() 

        if is_cross:
            attn = self.cross_attn_control(attn, place_in_unet)
        else:
            attn = self.self_attn_control(attn, place_in_unet)

        if self.cached_attn_info_flag:
            self.cached_attn_name_set.add(place_in_unet)
            if is_cross and self.do_cross_attn_control_flag:
                self.set_cached_attn(place_in_unet, is_cross, attn)
            elif is_cross and not self.do_cross_attn_control_flag:
                self.set_cached_attn(place_in_unet, is_cross, None)
            elif not is_cross and self.do_self_attn_control_flag:
                self.set_cached_attn(place_in_unet, is_cross, attn)
            else:
                self.set_cached_attn(place_in_unet, is_cross, None)
        return attn
        
    @property
    def control_attn_name_ls(self):
        return list(set(self.cross_attn_name_ls) - self.not_control_attn_name_set)
    
    def cross_attn_control(self, attn, place_in_unet):
        return attn

    def self_attn_control(self, attn, place_in_unet):
        return attn


class EmptyController(BaseController):
    def __init__(
            self,
            max_vis_pixel_num=MAX_VIS_PIXEL_NUM,
            edit_params: Optional[EditParams] = None,
            logger=base_logger
    ):
        super(EmptyController, self).__init__(
            max_vis_pixel_num=max_vis_pixel_num, cached_attn_info_flag=False, edit_params=edit_params, logger=logger)


class CachedController(BaseController):
    def __init__(
            self, max_vis_pixel_num=MAX_VIS_PIXEL_NUM, edit_params: Optional[EditParams] = None, logger=base_logger
    ):
        super(CachedController, self).__init__(
            max_vis_pixel_num=max_vis_pixel_num, cached_attn_info_flag=True, edit_params=edit_params, logger=logger)
        print(f">>>cached_attn_flag: {self.cached_attn_info_flag}")
    

class EditController(BaseController):
    def __init__(
            self, edit_params: EditParams,
            max_vis_pixel_num: int = MAX_VIS_PIXEL_NUM,
            cached_attn_info_flag: bool = False,
            logger=base_logger
    ):
        super().__init__(
            max_vis_pixel_num=max_vis_pixel_num, 
            cached_attn_info_flag=cached_attn_info_flag, edit_params=edit_params, logger=logger)
        self.control_info_checking()
        
    def cross_attn_control(self, attn: torch.Tensor, place_in_unet: str) -> torch.Tensor:

        if self.do_cross_attn_control_flag:
            source_replace_mask: torch.Tensor = self.replace_index_map["source_mask"]
            target_replace_mask: torch.Tensor = self.replace_index_map["target_mask"]
            source_token_weight = self.replace_index_map["source_token_weight"].squeeze(dim=0)
            target_token_weight = self.replace_index_map["target_token_weight"].squeeze(dim=0)
            cur_source_attn: torch.Tensor = self.get_source_attn(place_in_unet)  # "(b h) p c"
            assert target_token_weight.ndim == attn.ndim == source_token_weight.ndim == cur_source_attn.ndim

            attn = attn * target_token_weight
            cur_source_attn = cur_source_attn * source_token_weight
            attn[:, :, target_replace_mask] = cur_source_attn[:, :, source_replace_mask]
            # source_replace_mask = source_replace_mask.squeeze(dim=0)
            # target_replace_mask = target_replace_mask.squeeze(dim=0)
            # attn = attn * target_replace_mask + cur_source_attn * source_replace_mask
            if self.do_local_blend and self.text_branch_flag:  # local blend whatever cross control
                blend_attn = torch.cat([cur_source_attn, attn], dim=0)
                self.set_blend_attn_map(place_in_unet, True, blend_attn)
        return attn
    
    def self_attn_control(self, attn: torch.Tensor, place_in_unet: str) -> torch.Tensor:
        if attn.shape[2] <= 16 ** 2:
            if self.do_self_attn_control_flag:
                attn = self.get_source_attn(place_in_unet)
        return attn


class EditControllerMemEfficient(BaseController):
    def __init__(
            self, edit_params: EditParams,
            max_vis_pixel_num=MAX_VIS_PIXEL_NUM,
            cached_attn_info_flag=False,
            logger=base_logger
    ):
        super(EditControllerMemEfficient, self).__init__(
            max_vis_pixel_num=max_vis_pixel_num, 
            cached_attn_info_flag=cached_attn_info_flag, edit_params=edit_params, logger=logger)
        self.control_info_checking()

    def cross_attn_control(self, attn: torch.Tensor, place_in_unet: str) -> torch.Tensor:
        assert attn.shape[0] > 1, f"attn shape: {attn.shape}"
        source_replace_mask = self.replace_index_map["source_mask"]
        target_replace_mask = self.replace_index_map["target_mask"]
        source_token_weight = self.replace_index_map["source_token_weight"]
        target_token_weight = self.replace_index_map["target_token_weight"]

        if self.do_cross_attn_control_flag:
            attn = rearrange(attn, "(b h) p c -> b h p c", b=self.batch_size)
            source_attn = attn[:1, ...]
            target_attn = attn[1:, ...]

            source_attn_for_merge = source_attn * source_token_weight
            target_attn = target_attn * target_token_weight
            target_attn[..., target_replace_mask] = source_attn_for_merge[..., source_replace_mask]
            attn = torch.cat([source_attn, target_attn], dim=0)

            attn = rearrange(attn, "b h p c -> (b h) p c")

        if self.do_local_blend and self.text_branch_flag:  # local blend whatever cross control
            blend_attn = attn
            self.set_blend_attn_map(place_in_unet, True, blend_attn)
        return attn
    
    def self_attn_control(self, attn: torch.Tensor, place_in_unet: str) -> torch.Tensor:
        if attn.shape[2] <= 16 ** 2:
            attn = rearrange(attn, "(b h) p c -> b h p c", b=self.batch_size)
            source_attn = attn[:1, ...]
            if self.do_self_attn_control_flag:
                attn = source_attn.expand(self.batch_size, *source_attn.shape[1:])
            attn = rearrange(attn, "b h p c -> (b h) p c")
        return attn
    

def control_cross_attn_forward(self, controller: BaseController, place_in_unet):
    def forward(x, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        batch_size, sequence_length, dim = x.shape
        h = self.heads
        q = self.to_q(x)
        is_cross = context is not None
        context = context if is_cross else x
        k = self.to_k(context)
        v = self.to_v(context)
        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            mask = mask.reshape(batch_size, -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = mask[:, None, :].repeat(h, 1, 1)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        # print(f"attn shape: {attn.shape}")
        attn = controller.control(attn, is_cross, place_in_unet)  # AttentionStore时相当于将attention值缓存到controller中
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = self.reshape_batch_dim_to_heads(out)
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]  # 忽略dropout
        else:
            to_out = self.to_out
        return to_out(out)
    return forward


def register_attention_control_mine(unet, controller):
    cross_attn_name_ls = []
    for i in unet.named_children():
        name, cur_module = i[:2]
        if cur_module.__class__.__name__ == "CrossAttention":
            cur_module.forward = control_cross_attn_forward(cur_module, controller, name)
            cross_attn_name_ls.append(name)
        elif hasattr(cur_module, "children"):
            module_ls = [(name, cur_module)]
            while module_ls:
                name, cur_module = module_ls.pop()
                for sub_name, sub_module in cur_module.named_children():
                    if sub_module.__class__.__name__ == "CrossAttention":
                        sub_module.forward = control_cross_attn_forward(
                            sub_module,
                            controller,
                            f"{name}.{sub_name}"
                        )
                        cross_attn_name_ls.append(f"{name}.{sub_name}")
                    elif hasattr(sub_module, "children"):
                        module_ls.append((f"{name}.{sub_name}", sub_module))
    controller.num_att_layers = len(cross_attn_name_ls)
    controller.cross_attn_name_ls = cross_attn_name_ls


class BaseDiffusionModel:
    def __init__(self, seed=SEED, logger=base_logger):
        self.model: StableDiffusionPipeline = ldm_stable
        self.unet: UNet2DConditionModel = self.model.unet
        self.tokenizer: CLIPTokenizer = self.model.tokenizer
        self.vae: AutoencoderKL = self.model.vae
        self.text_encoder: CLIPTextModel = self.model.text_encoder
        # self.img_encoder: CLIPVisionModel = clip_img_encoder
        self.feature_extractor: CLIPFeatureExtractor = self.model.feature_extractor
        self.device = self.model.device
        self.scheduler: PNDMScheduler = self.model.scheduler
        self.logger = logger 
        self.seed = seed 

    def init_latent(
            self,
            batch_size: int,
            generator: torch.Generator,
            latent: Optional[torch.Tensor] = None,
            height: int = DEFAULT_IMG_HEIGHT,
            width: int = DEFAULT_IMG_WIDTH,
    ) -> torch.FloatTensor:
        """
        init latent noise mocked VAE's encoder output which downsample 8x 
        Args:
            batch_size: int
            latent: torch.Tensor, shape: (1, in_channels, height // 8, width // 8)
            height: int
            width: int
            generator: torch.Generator
        Returns:
            latents: torch.Tensor, shape: (batch_size, in_channels, height // 8, width // 8)
        """
        if latent is None:
            latent = torch.randn(
                (1, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
                dtype=TORCH_DTYPE
            )
        latents = latent.expand(batch_size,  self.unet.in_channels, height // 8, width // 8).to(device=self.device, dtype=TORCH_DTYPE)

        return latents
    
    def image2image(self, image: List[np.ndarray]) -> List[np.ndarray]:
        # assert isinstance(image, list), f"image must be list, but received: {type(image)}"
        # images = np.stack(image, axis=0)
        # image_tensor = self.numpy_to_pil(images)
        # image_embed = self.get_image_embed(image)
        pass

    def get_classifier_free_prompt_embed(self, prompt: List[str]) -> List[torch.Tensor]:
        """

        Args:
            prompt:

        Returns:

        """
        prompt_embed = self.get_text_embed(prompt)
        uncond_embed = self.get_uncond_embedding(len(prompt))
        self.logger.debug(f"prompt_embed shape: {prompt_embed.shape}, uncond_embed shape: {uncond_embed.shape}")
        context = [uncond_embed, prompt_embed]
        return context

    @torch.no_grad()
    def text2image(self, prompt: List[str]) -> List[np.ndarray]:
        context = self.get_classifier_free_prompt_embed(prompt)
        latents = self.init_latent(len(prompt), generator=torch.Generator().manual_seed(self.seed))
        self.logger.debug(f"init latent shape: {latents.shape}")
        self.scheduler.set_timesteps(NUM_DIFFUSION_STEPS)
        for t in tqdm(self.scheduler.timesteps):
            latents = self.diffusion_step(latents, context, t, guidance_scale=GUIDANCE_SCALE, low_resource=LOW_RESOURCE)
        image = self.latent2image(latents)
        return [image[i] for i in range(image.shape[0])]

    def get_text_token_id(self, prompt: List[str]) -> torch.Tensor:
        """
        given prompt, return text token id
        Example:
            prompt = ["a photo of a cat", "a photo of a dog"]
            text_token_id = get_text_token_id(prompt)  # text_token_id.shape: (2, MAX_NUM_WORDS)
        Args:
            prompt: List[str]
        Returns:
            text_token_id: torch.Tensor, shape: (batch_size, max_length)
        """
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=MAX_NUM_WORDS,
            truncation=True,
            return_tensors="pt",
        )
        text_token_id = text_input.input_ids.to(self.device)
        return text_token_id
    
    @torch.no_grad()
    def get_image_embed(self):
        pass 

    @torch.no_grad()
    def get_text_embed(self, prompt: List[str]) -> torch.Tensor:
        """
        given prompt, return text embedding
        Example:
            prompt = ["a photo of a cat", "a photo of a dog"]
            text_embed = get_text_embed(prompt)  # text_embed.shape: (2, MAX_NUM_WORDS, dim)   
        Args:
            prompt: List[str]
        Returns:
            text_embed: torch.Tensor, shape: (batch_size, max_length, dim)
        """
        text_token_id = self.get_text_token_id(prompt)
        text_embeddings: torch.Tensor = self.text_encoder(text_token_id)[0]
        return text_embeddings.type(TORCH_DTYPE)
    
    @torch.no_grad()
    def get_uncond_embedding(self, batch_size: int) -> torch.Tensor:
        """
        get unconditional embedding
        Args:
            batch_size: int
        Returns:
            uncond_embeddings: torch.Tensor, shape: (batch_size, max_length, dim)
        """ 
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=MAX_NUM_WORDS, return_tensors="pt"
        )
        uncond_embeddings: torch.Tensor = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        # shape: (1, 77, 768) [batch_size, max_length, dim]
        return uncond_embeddings.type(TORCH_DTYPE)
    
    @torch.no_grad()
    def diffusion_step(self, latents, context, t, guidance_scale=GUIDANCE_SCALE, low_resource=True) -> torch.FloatTensor:
        """
        diffusion step
        Args:
            latents: torch.Tensor, shape: (batch_size, in_channels, height // 8, width // 8)
            context: List[Tensor], (context_uncond, context_text), each shape: (batch_size, MAX_NUM_WORDS, dim)
            t: int, current diffusion step
            guidance_scale: float
            low_resource: bool
        Returns:
            latents: torch.FloatTensor, shape: (batch_size, in_channels, height // 8, width // 8)
        """
        if low_resource:  # 算力有限时分开算
            noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            noise_prediction_text = self.unet(latents, t, encoder_hidden_states=context[1])["sample"]
        else:
            context = torch.cat(context, dim=0)
            latents_input = torch.cat([latents] * 2)
            noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        return latents

    @torch.no_grad()
    def latent2image(self, latents: torch.Tensor) -> np.ndarray:
        """
        decode latent to image
        """
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images


class Prompt2Prompt(BaseDiffusionModel):
    def __init__(self, seed=SEED, logger=base_logger):
        super(Prompt2Prompt, self).__init__(seed=seed, logger=logger)

    def get_context_embed(self, prompt_ls: List[str]) -> List[torch.Tensor]:
        """
        given prompt, return context embedding. the first item is unconditional embedding,
        the second item is prompt embedding
        Args:
            prompt_ls (List[str]): prompt
        Returns:
            context (List[torch.Tensor]): first item is unconditional embedding,
            the second item is prompt embedding
        """
        prompt_embed = self.get_text_embed(prompt_ls)
        uncond_embed = self.get_uncond_embedding(len(prompt_ls))
        self.logger.debug(f"prompt_embed shape: {prompt_embed.shape}, uncond_embed shape: {uncond_embed.shape}")
        context = [uncond_embed, prompt_embed]
        return context

    @torch.no_grad()
    def text2image(
            self, 
            prompt: Union[List[str], str],
            latent: Optional[torch.FloatTensor] = None,
            cached_attn_flag: bool = False, 
            edit_params: Optional[EditParams] = None
    ) -> Text2ImageResult:
        if isinstance(prompt, str):
            prompt = [prompt]
        if cached_attn_flag:
            controller = CachedController()
        else:
            controller = EmptyController()
        controller.init_edit_params(edit_params)
        register_attention_control_mine(self.unet, controller)

        context = self.get_classifier_free_prompt_embed(prompt=prompt)

        latents: torch.FloatTensor = self.init_latent(
            len(prompt),
            generator=torch.Generator().manual_seed(self.seed),
            latent=latent
        )
        result = Text2ImageResult(image=None, cached_attn_with_step=None, latent=latents)
        self.logger.debug(f"init latent shape: {latents.shape}")
        cached_attn_with_step = []
        cached_latent_with_step = []
        self.scheduler.set_timesteps(NUM_DIFFUSION_STEPS)
        total_step = len(self.scheduler.timesteps)
        for cur_step, t in tqdm(enumerate(self.scheduler.timesteps), total=total_step):
            latents = self.diffusion_step(
                latents, context, t, cur_step,
                guidance_scale=GUIDANCE_SCALE, low_resource=LOW_RESOURCE, controller=controller)
            cached_attn_with_step.append(controller.cached_attn)
            cached_latent_with_step.append(latents.detach().cpu())
            if cur_step < total_step - 1:
                controller.reset_cached_info()
        image = self.latent2image(latents)
        result.image = image
        result.cached_attn_with_step = cached_attn_with_step
        result.cached_latent_with_step = cached_latent_with_step
        return result

    @torch.no_grad()
    def text2image_with_control(
            self,
            edit_params: EditParams,
            source_text2img_result: Optional[Text2ImageResult] = None,
            latent: Optional[torch.FloatTensor] = None,
            cached_attn_flag: bool = False,
            cached_latent_flag: bool = False,
            cached_source_flag: bool = False
    ) -> ImageEditResult:
        if source_text2img_result is None:
            source_text2img_result = self.text2image(
                [edit_params.source_prompt],
                latent=latent,
                cached_attn_flag=True,
                edit_params=edit_params
            )
        else:
            if source_text2img_result.cached_attn_with_step is None:
                source_text2img_result = self.text2image(
                    [edit_params.source_prompt],
                    latent=latent,
                    cached_attn_flag=True,
                    edit_params=edit_params
                )
            else:
                if edit_params.do_local_blend and source_text2img_result.cached_latent_with_step is None:
                    source_text2img_result = self.text2image([edit_params.source_prompt], cached_attn_flag=True,
                                                             edit_params=edit_params)

        edit_controller = EditController(edit_params, cached_attn_info_flag=cached_attn_flag)

        register_attention_control_mine(self.unet, edit_controller)
        source_cached_attn_with_step = source_text2img_result.cached_attn_with_step
        source_latent_with_step = source_text2img_result.cached_latent_with_step

        context = self.get_context_embed([edit_params.target_prompt])
        latents: torch.FloatTensor = source_text2img_result.latent
        self.logger.debug(f"init latent shape: {latents.shape}")
        cached_attn_with_step = []
        cached_latent_with_step = []

        edit_controller.set_batch_size(latents.shape[0])
        self.scheduler.set_timesteps(NUM_DIFFUSION_STEPS)
        total_step = len(self.scheduler.timesteps)
        for cur_step, t in tqdm(enumerate(self.scheduler.timesteps), total=total_step):
            if not cached_source_flag:
                cur_source_cached_attn = source_cached_attn_with_step.pop(0)
                cur_source_latent = source_latent_with_step.pop(0)
            else:
                cur_source_cached_attn = source_cached_attn_with_step[cur_step]
                cur_source_latent = source_latent_with_step[cur_step]

            latents = self.diffusion_step(
                latents, context, t, cur_step,
                guidance_scale=GUIDANCE_SCALE,
                low_resource=True,
                source_cached_attn=cur_source_cached_attn,
                controller=edit_controller
            )
            if edit_params.do_local_blend:
                attn_map = edit_controller.blend_attn_map
                cur_source_latent = cur_source_latent.to(device=self.device, dtype=TORCH_DTYPE)
                if len(attn_map) > 0:
                    latents = edit_controller.local_blend_obj(
                        torch.cat([cur_source_latent, latents], dim=0),
                        edit_params.blend_token_map,
                        attn_map
                    )[1:, ...]
                else:
                    self.logger.debug(f"attn map have not cached! please check!")

            if cached_attn_flag:
                cached_attn_with_step.append(edit_controller.cached_attn)
            if cached_latent_flag:
                cached_latent_with_step.append(latents.detach().cpu())
            edit_controller.reset_cached_info()

        image = self.latent2image(latents)
        return ImageEditResult(
            target_image=image, 
            source_image=source_text2img_result.image,
            latent=source_text2img_result.latent,
            cached_attn_with_step=cached_attn_with_step if len(cached_attn_with_step) > 0 else None,
            cached_latent_with_step=cached_latent_with_step if len(cached_latent_with_step) > 0 else None,
            cached_source_result=source_text2img_result if cached_source_flag else None,
            )

    @torch.no_grad()
    def text2image_with_control_memory_saving(
            self,
            edit_params: EditParams,
            latent: Optional[torch.FloatTensor] = None,
            cached_attn_flag: bool = False,
            cached_latent_flag: bool = False,
            **kwargs
    ) -> ImageEditResult:

        edit_controller = EditControllerMemEfficient(edit_params, cached_attn_info_flag=False)
        register_attention_control_mine(self.unet, edit_controller)
        merge_prompt = extend_list(edit_params.source_prompt, edit_params.target_prompt)
        context = self.get_context_embed(merge_prompt)
        latents: torch.FloatTensor = self.init_latent(
            context[0].shape[0],
            torch.Generator().manual_seed(self.seed),
            latent=latent
        )
        latent = latents
        edit_controller.set_batch_size(latents.shape[0])
        self.logger.debug(f"init latent shape: {latents.shape}")
        self.scheduler.set_timesteps(NUM_DIFFUSION_STEPS)
        total_step = len(self.scheduler.timesteps)
        with tqdm(self.scheduler.timesteps, total=total_step) as pbar:
            for cur_step, t in enumerate(pbar):
                latents = self.diffusion_step(
                    latents, context, t, cur_step,
                    guidance_scale=GUIDANCE_SCALE,
                    low_resource=True,
                    controller=edit_controller
                )
                if edit_params.do_local_blend:
                    attn_map = edit_controller.blend_attn_map
                    if len(attn_map) > 0:
                        pbar.set_description("edit with local blend")
                        latents = edit_controller.local_blend_obj(latents, edit_params.blend_token_map, attn_map)
                    else:
                        pbar.set_description("edit without local blend")
                edit_controller.reset_cached_info()
            image = self.latent2image(latents)
            return ImageEditResult(
                source_image=image[:1, ...],
                target_image=image[1:, ...],
                latent=latent,
                cached_attn_with_step=None,
                cached_latent_with_step=None,
                cached_source_result=None
            )

    @torch.no_grad()
    def diffusion_step(
            self,
            latents: torch.FloatTensor,
            context: List[torch.Tensor],
            t: int,
            cur_step: int,
            guidance_scale: float = GUIDANCE_SCALE,
            low_resource: bool = True,
            source_cached_attn: Optional[Dict] = None,
            controller: Optional[BaseController] = None
    ):
        """
        diffusion step
        Args:
            latents: torch.Tensor, shape: (batch_size, in_channels, height // 8, width // 8)
            context: List[Tensor], (context_uncond, context_text), each shape: (batch_size, MAX_NUM_WORDS, dim)
            t: int, current diffusion time step
            cur_step: current iteration step
            guidance_scale: float
            low_resource: bool
            source_cached_attn: Optional[Dict], cached attention
            controller: BaseController
        Returns:
            latents: torch.Tensor, shape: (batch_size, in_channels, height // 8, width // 8)
        """
        if controller is not None:
            controller.set_step_source_attn_map(source_cached_attn)
            controller.set_step(cur_step)
        if low_resource:  # 算力有限时分开算
            if controller is not None:
                controller.set_text_branch_flag(False)
            noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            if controller is not None:
                controller.set_text_branch_flag(True)
            noise_prediction_text = self.unet(latents, t, encoder_hidden_states=context[1])["sample"]
        else:
            # context = torch.cat(context, dim=0)
            # latents_input = torch.cat([latents] * 2)
            # noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            # noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            raise NotImplementedError
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        return latents


@torch.no_grad()
def text2image(
        prompt: Optional[List[str]] = None,
        save_path: Optional[str] = None, 
        cached_attn_flag: bool = False,
        seed=SEED
        ) -> Text2ImageResult:
    p2p_obj = Prompt2Prompt(seed=seed)
    res = p2p_obj.text2image(prompt=prompt, cached_attn_flag=cached_attn_flag)
    if save_path is not None:
        Image.fromarray(res.image[0]).save(save_path) 
    return res


@torch.no_grad()
def text2image_with_control(
        edit_params: EditParams,
        memory_saving: bool = False,
        save_rtpath: Optional[str] = None,
        latent: Optional[torch.Tensor] = None,
        source_text2img_result: Optional[Text2ImageResult] = None,
        cached_attn_flag=False,
        cached_latent_flag: bool = False,
        cached_source_flag: bool = False,
        seed: int = SEED
        ) -> ImageEditResult:
    """

                source_text2img_result: Optional[Text2ImageResult] = None,
                cached_attn_flag: bool = False,
                cached_latent_flag: bool = False,
                cached_source_flag: bool = False
    """
    p2p_obj = Prompt2Prompt(seed=seed)

    if not memory_saving:
        text2img_with_control_func = p2p_obj.text2image_with_control
    else:
        text2img_with_control_func = p2p_obj.text2image_with_control_memory_saving

    res: ImageEditResult = text2img_with_control_func(
        edit_params=edit_params,
        latent=latent,
        source_text2img_result=source_text2img_result,
        cached_attn_flag=cached_attn_flag,
        cached_latent_flag=cached_latent_flag,
        cached_source_flag=cached_source_flag
    )
    source_image = res.source_image
    target_image = res.target_image
    if save_rtpath is not None:
        checkdir(save_rtpath)
        Image.fromarray(source_image[0]).save(os.path.join(save_rtpath, f"source.png"))
        Image.fromarray(target_image[0]).save(os.path.join(save_rtpath, f"target.png"))
    torch.cuda.empty_cache()
    return res


def draw_attn_exp(save_dir_name, seed=SEED, aim_shape=(64, 64)):
    exp_ls = [
        dict(
            prompt="A painting of a squirrel eating a burger",
            cached_attn_flag=True,
            seed=seed
        ),
        dict(
            prompt="a photo of a house on a mountain.",
            cached_attn_flag=True,
            seed=seed
        ),
        dict(
            prompt="A of a squirrel eating a burger",
            cached_attn_flag=True,
            seed=seed
        ),
        dict(
            prompt="pink bear riding a bicycle",
            cached_attn_flag=True,
            seed=seed
        )

    ]
    for exp_param in tqdm(exp_ls):
        cur_token_ls = [tokenizer.decode(i) for i in tokenizer.encode(exp_param["prompt"])]
        res = text2image(**exp_param)
        cur_cached_attn_with_step: List[Dict[str, Dict[str, CachedAttnInfo]]] = res.cached_attn_with_step
        attn_info: Dict[str, List[Dict[str, np.ndarray]]] = draw_cross_attention_per_layer_all_time_step(
            cur_token_ls,
            cur_cached_attn_with_step,
            aim_shape=aim_shape
        )
        for cur_place, cur_place_attn_info in attn_info.items():
            pass


if __name__ == "__main__":
    p2p_obj = Prompt2Prompt(seed=SEED)
    edit_params = EditParams(
        source_prompt="a photo of a house on a mountain.",
        target_prompt=["a photo of a house on a mountain at fall", "a photo of a house on a mountain at winter"],
        cross_merge_start_step=0,
        cross_merge_end_step=0.6,
        self_merge_start_step=0,
        self_merge_end_step=0.2,
        addition_token_control_info=None,
        do_noise_branch_control=False,
        do_local_blend=False,
        blend_focus_text=None,
    )

    res = p2p_obj.text2image_with_control(edit_params)
    
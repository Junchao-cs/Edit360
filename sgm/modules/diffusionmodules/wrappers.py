import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


# class OpenAIWrapper(IdentityWrapper):
#     def forward(
#         self, x: torch.Tensor, t: torch.Tensor, c: dict, skip_input = None, **kwargs
#     ) -> torch.Tensor:
#         x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
#         if "cond_view" in c:
#             return self.diffusion_model(
#                 x,
#                 timesteps=t,
#                 context=c.get("crossattn", None),
#                 y=c.get("vector", None),
#                 cond_view=c.get("cond_view", None),
#                 cond_motion=c.get("cond_motion", None),
#                 **kwargs,
#             )
#         else:
#             return self.diffusion_model(
#                     x,
#                     timesteps=t,
#                     context=c.get("crossattn", None),
#                     y=c.get("vector", None),
#                     **kwargs,
#                 )


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, is_front_view, kv_f_list, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        if "cond_view" in c:
            return self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                cond_view=c.get("cond_view", None),
                cond_motion=c.get("cond_motion", None),
                **kwargs,
            )
        else:
            if is_front_view == True:
                output, kv_f_list = self.diffusion_model(
                    x,
                    timesteps=t,
                    context=c.get("crossattn", None),
                    y=c.get("vector", None),
                    is_front_view=is_front_view,
                    kv_f_list=kv_f_list,
                    **kwargs,
                )
                return output, kv_f_list
            else:
                output, kv_f_list = self.diffusion_model(
                    x,
                    timesteps=t,
                    context=c.get("crossattn", None),
                    y=c.get("vector", None),
                    is_front_view=is_front_view,
                    kv_f_list=kv_f_list,
                    **kwargs,
                )
                return output, kv_f_list

            

# class OpenAIWrapper(IdentityWrapper):
#     def forward(
#         self, x: torch.Tensor, t: torch.Tensor, c: dict, skip_input = None, **kwargs
#     ) -> torch.Tensor:
#         x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
#         if "cond_view" in c:
#             return self.diffusion_model(
#                 x,
#                 timesteps=t,
#                 context=c.get("crossattn", None),
#                 y=c.get("vector", None),
#                 cond_view=c.get("cond_view", None),
#                 cond_motion=c.get("cond_motion", None),
#                 **kwargs,
#             )
#         else:
#             if skip_input is None:
#                 output, skip_output = self.diffusion_model(
#                     x,
#                     timesteps=t,
#                     context=c.get("crossattn", None),
#                     y=c.get("vector", None),
#                     **kwargs,
#                 )
#                 return output, skip_output
#             else:
#                 output = self.diffusion_model(
#                     x,
#                     timesteps=t,
#                     context=c.get("crossattn", None),
#                     y=c.get("vector", None),
#                     skip_input=skip_input,
#                     **kwargs,
#                 )
#                 return output

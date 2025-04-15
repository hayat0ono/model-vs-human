import os
from modelvshuman import models


def add_my_model(model_root):
    available_model_lists = models.list_models("pytorch")

    model_names = []
    for root, dirs, files in os.walk(model_root):
        if 'model.pth' in files:
            model_name = os.path.relpath(root, model_root)
            model_names.append(model_name)

    for model_name in model_names:
        if model_name.replace('/', '_').replace('.', '') not in available_model_lists:
            new_code = f'''
@register_model("pytorch")
def {model_name.replace('/', '_').replace('.', '')}(model_name, *args):
    from .my_model.load_my_model import get_local_model
    model = get_local_model('{model_name.replace('/', '-')}', '/home/ono/PycharmProjects/proj_metamer_cornet/model/', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return PytorchModel(model, model_name, *args)
'''
            with open('/home/ono/PycharmProjects/model-vs-human/modelvshuman/models/pytorch/model_zoo.py', 'a') as f:
                f.write('\n' + new_code)
import os
import time

import src.model_manager as m
import src.evaluate as e
import constants as c

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


m.add_my_model(c.MODEL_ROOT)
time.sleep(10)
my_models = m.get_my_model_names(c.MODEL_ROOT)
my_model_to_model_name = {}
for my_model in my_models:
    my_model_to_model_name[my_model] = my_model.replace('-', '_').replace('.', '')
e.evaluate(my_model_to_model_name.values())
for my_model in my_models:
    metrics = m.get_metrics(c.DEFAULT_METRICS, c.MODEL_ROOT, my_model)
    metric_dict = e.get_metrics(metrics, my_model_to_model_name[my_model], c)
    m.save_metrics(c.MODEL_ROOT, my_model, metric_dict)
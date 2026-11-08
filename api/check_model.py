"""Run this FIRST:  python check_model.py
Confirms your week6_model.pt loads and shows the settings app.py needs."""
import torch

ck = torch.load('week6_model.pt', map_location='cpu', weights_only=False)
print('Keys     :', sorted(ck.keys()))
print('Backbone :', ck['backbone'])
print('Classes  :', ck['classes'])
print('Grayscale:', ck['grayscale'])
print('Img size :', ck['img_size'])
print('\nSet POSITIVE in app.py to one of the class names above (the class you must not miss).')

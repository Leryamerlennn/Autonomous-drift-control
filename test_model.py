import pytest
import torch


################################################################################
# 3. Neural net definition
################################################################################
class DynNet(torch.nn.Module):
    def __init__(self):
        super(DynNet, self).__init__()
        self.fc1 = torch.nn.Linear(6, 128)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 128)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(128, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Функция для исправления ключей
def fix_state_dict_keys(state_dict):
    # Преобразуем ключи, например "0.weight" -> "fc1.weight"
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net.0."):
            new_key = k.replace("net.0.", "fc1.")
        elif k.startswith("net.2."):
            new_key = k.replace("net.2.", "fc2.")
        elif k.startswith("net.4."):
            new_key = k.replace("net.4.", "fc3.")
        else:
            new_key = k
        fixed_state_dict[new_key] = v
    return fixed_state_dict


@pytest.fixture
def model():
    # Loading the model
    # Загрузка состояния модели
    checkpoint = torch.load('ML/dyn_v3.pt', map_location='cpu', weights_only=False)

    # Обрезаем префикс и адаптируем ключи
    state_dict = checkpoint['net']
    state_dict = fix_state_dict_keys(state_dict)

    # Создаем модель и загружаем в нее исправленные веса
    model = DynNet()
    model.load_state_dict(state_dict)

    model.eval()  # Переводим модель в режим инференса
    return model

def test_model_loading(model):
    # Проверяем, что модель загружается без ошибок
    assert model is not None, "Model failed to load"

def test_model_output(model):
    # Проверим, что модель выдает выходные данные для случайного входа
    input_data = torch.randn(1, 6)  # Предположим, что модель принимает 1 входной вектор размера 6
    output = model(input_data)
    assert output is not None, "Model output is None"
    assert output.size(0) == 1, "Model output batch size is incorrect"

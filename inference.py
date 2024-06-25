import torch
import torchvision.transforms as transforms

from data_model import ExampleDataset
from model import ExampleModel
from visualization import preprocess_image, visualize_predictions


data_dir = "./model_dataset/train"
model_path = "./model/animal_7.pth"
test_image_path = "model_dataset/test/bird/image_8.jpg"
image_size = (128, 128)
num_classes = 4


def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
print(device)

model = ExampleModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()
dataset = ExampleDataset(data_dir)

original_image, image_tensor = preprocess_image(test_image_path, transform)
probabilities = predict(model, image_tensor, device)

class_name = dataset.classes
visualize_predictions(original_image, probabilities, class_name)

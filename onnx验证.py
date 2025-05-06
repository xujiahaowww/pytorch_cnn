import cv2
import onnx
import onnxruntime as ort
from torchvision import transforms

transformdata = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((100, 100)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def imgread(img_path):
    imgdata = cv2.imread(img_path)
    # 转换成RGB
    imgdata = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    imgdata = transformdata(imgdata)
    # tensor ---> CHW  --->NCHW
    imgdata = imgdata.unsqueeze(0).numpy()
    return imgdata


def inference():
    # 加载onnx模型
    model = ort.InferenceSession(
        "./kaggle.onnx", providers=["CPUExecutionProvider"]
    )
    imgdata = imgread("./test3.jpg")
    out = model.run(None, {"input": imgdata})
    print(out)
    classlabels = ["daisy", "rose ", "tulip", "sunflower", "dandelion"]
    print(classlabels[list(out[0][0]).index(max(out[0][0]))])


if __name__ == "__main__":
    inference()
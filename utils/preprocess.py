from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path: str) -> Image:
    """
    画像を開き、前処理を行う関数。
    
    :param image_path: 画像ファイルのパス
    :return: 前処理された画像
    """
    image = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # 画像のリサイズ
        transforms.CenterCrop(224),     # センタクロップ
        transforms.ToTensor(),          # テンソルに変換
        transforms.Normalize(           # 正規化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = preprocess(image)
    return image
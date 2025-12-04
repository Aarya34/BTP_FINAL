# Zero-Shot Domain Adaptation Object Detection


## 1. Project Structure

Please ensure your folders are organized exactly as shown below for the commands to work without errors:

```text
BTP_FINAL/
│
├── data/
│   ├── coco/
│   │   ├── annotations/
│   │   │   └── instances_train2017.json   <-- Source Annotations
│   │   └── train2017/                     <-- Source Images
│   └── clipart1k/                         <-- Target Images (No Labels)
│
├── test_images/                           <-- PUT INPUT IMAGES HERE
├── test_results/                          <-- OUTPUT SAVED HERE
├── train.py
├── inference_rpn.py
├── model.py
├── datasets.py
└── utils.py
```
## 2. Installation

Install the required Python dependencies:

```text
pip install torch torchvision pillow
```

## 3. Training the Model
To train the model adapting from COCO to Clipart, run the following command.

```text
python train.py --src-root data/coco/train2017 --src-anns data/coco/annotations/instances_train2017.json --tgt-root data/clipart1k --batch-size 2 --epochs 10
```

## 4. Testing (Inference)
To detect objects in new images:
1. Create a folder named test_images and put your images (jpg/png) inside it.
2. Run the command below using the checkpoint saved from the last epoch (e.g., checkpoint_epoch9.pth).
```text
python inference_rpn.py --ckpt checkpoint_epoch9.pth --input-folder test_images --output-folder test_results --score-thresh 0.5
```
The resulting images with drawn bounding boxes will be saved to the test_results folder.

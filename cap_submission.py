from functools import partial
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from nltk.translate.bleu_score import corpus_bleu
from transformers import Seq2SeqTrainer, VisionEncoderDecoderModel, AutoProcessor, GPT2Tokenizer
from transformers import Seq2SeqTrainingArguments, default_data_collator

#################
#   IMPORTANT!! #
#   train.txt and val.txt and the images folder with all the training images should all be in a folder together called flickr8k or the code will not be able to access them properly #        


# ##########
# TODO: Add more imports
import json
# ##########
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(f'Cuda Device Count is {torch.cuda.device_count()}')

class Args:
    """Configuration.
    """
    # Encoder-Decoder for captioning
    encoder = None
    decoder = None

    # Dataset path
    root_dir = "./flickr8k"
    img_folder = os.path.join(root_dir, "images")
    train_txt = os.path.join(root_dir, "train.txt")
    val_txt = os.path.join(root_dir, "val.txt")

    # Save your model as "cap-vlm-{YOUR_CCID}"
    YOUR_CCID = "asajid2"
    name = f"cap-vlm-{YOUR_CCID}"

    # Hyperparameters
    batch_size = 16
    lr = 5e-5
    epochs = 5

    # Generation cfgs
    num_beams = 5
    max_length = 45

    # Train ops
    logging_steps = 50

class FlickrDataset(Dataset):
    def __init__(
        self, 
        args, 
        processor, 
        tokenizer,
        mode: str = "train",
        ):
        assert mode in ["train", "val"]
        self.args = args
        self.processor = processor
        self.tokenizer = tokenizer
        txt_file = args.train_txt if mode == "train" else args.val_txt

        # ####################
        # TODO: Load Flickr8k dataset
        self.img_paths = []
        self.captions = []

        with open(txt_file, 'r') as f:
            next(f)
            for line in f:
                img, caption = line.strip().split(';')
                self.img_paths.append(os.path.join(args.img_folder, img))
                self.captions.append(caption)
        # ####################

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        caption = self.captions[idx]
        
        # print(f"Attempting to load image: {img_path}")  # Add this line to see the constructed image path
        
        # If processor or tokenizer are None, return the raw data (used for `inspect_data`)
        if self.processor is None or self.tokenizer is None:
            return {
                "path": img_path,
                "captions": caption,
                "labels": None  # Include this to avoid KeyError
            }

        # Load and process image when processor is available
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        # Process caption (tokenization and padding)
        labels = self.tokenizer(caption, padding="max_length", max_length=self.args.max_length, return_tensors="pt").input_ids

        encoding = {
            "pixel_values": pixel_values.squeeze(),  # Return processed image as a tensor
            "labels": labels.squeeze(),             # Return tokenized caption as a padded tensor
            "path": img_path,
            "captions": caption,
        }

        return encoding


def train_cap_model(args):
    # Define your vision processor and language tokenizer
    processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Modify tokenizer to handle special tokens
    tokenizer.add_special_tokens({'bos_token': '<|beginoftext|>', 'pad_token': '<|pad|>', 'eos_token': '<|endoftext|>'})

    # Define your Image Captioning model
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    # Update the model's tokenizer embeddings
    model.decoder.resize_token_embeddings(len(tokenizer))

    if torch.cuda.is_available():
        model.cuda()

    # Load train/val dataset
    train_dataset = FlickrDataset(args, processor=processor, tokenizer=tokenizer, mode="train")
    val_dataset = FlickrDataset(args, processor=processor, tokenizer=tokenizer, mode="val")

    # Model generation configuration
    model.generation_config.max_length = args.max_length
    model.generation_config.num_beams = args.num_beams

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        predict_with_generate=True,
    )

    # Instantiate seq2seq model trainer
    compute_metrics = partial(compute_bleu_score, tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    # Start training
    trainer.train()
    trainer.save_model(args.name)

def load_trained_model(ckpt_dir: str):
    """Load your best trained model, processor, and tokenizer."""
    # Load model configuration
    model = VisionEncoderDecoderModel.from_pretrained(ckpt_dir)
    processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'bos_token': '<|beginoftext|>', 'pad_token': '<|pad|>', 'eos_token': '<|endoftext|>'})

    if torch.cuda.is_available():
        model.cuda()

    return model, processor, tokenizer

def inference(img_path, model, processor, tokenizer):
    """Example inference function to predict a caption for an image."""
    # Load and process the image
    image = Image.open(img_path).convert("RGB")
    img_tensor = processor(images=image, return_tensors="pt").pixel_values

    # Ensure img_tensor is on GPU
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    # Generate the caption
    generated_ids = model.generate(img_tensor, max_length=45, num_beams=5)
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_caption

def compute_bleu_score(pred, tokenizer):
    """Compute BLEU score."""
    pred_ids = pred.predictions
    labels_ids = pred.label_ids

    # Decode predictions and labels while handling special tokens and padding
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Prepare data for BLEU score calculation
    pred_bleu = [line.split() for line in pred_str]
    label_bleu = [[line.split()] for line in label_str]

    # Calculate BLEU score
    bleu_output = corpus_bleu(label_bleu, pred_bleu)
    bleu_score = round(bleu_output, 4)
    print("BLEU:", bleu_score)

    return {
        "bleu_score": bleu_score
    }

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torchvision.utils import save_image
from PIL import Image

# Function to load an image and convert it to a tensor (keeping original resolution)
def load_image(image_path, shape=None):
    image = Image.open(image_path).convert("RGB")
    
    # Define a transform to convert the image to a tensor (without resizing)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.cuda() if torch.cuda.is_available() else image

# Define your VGG model for extracting features
class VGGFeatures(torch.nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:29]  # Use up to layer 29
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.vgg(x)

# Loss functions for style transfer
def calculate_content_loss(target, content):
    return torch.mean((target - content) ** 2)

def calculate_style_loss(target, style_gram):
    target_gram = gram_matrix(target)
    return torch.mean((target_gram - style_gram) ** 2)

def gram_matrix(tensor):
    if len(tensor.size()) == 3:  # If the batch dimension is missing, add it
        tensor = tensor.unsqueeze(0)
    
    _, c, h, w = tensor.size()  # Ensure we have 4 dimensions
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (c * h * w)

# Style transfer function
def style_transfer(content_image_path, style_image_path, output_image_path, iterations=1000, content_weight=1e3, style_weight=1e4):
    # Load content and style images (keeping original resolution)
    content_img = load_image(content_image_path)
    style_img = load_image(style_image_path, shape=content_img.shape[-2:])
    
    # Initialize the target image as a clone of content
    target_img = content_img.clone().requires_grad_(True)
    
    # Load the VGG19 model
    vgg = VGGFeatures().cuda() if torch.cuda.is_available() else VGGFeatures()
    
    # Extract features from content and style images
    content_features = vgg(content_img)
    style_features = vgg(style_img)
    
    # Compute the style gram matrix
    style_grams = [gram_matrix(feature) for feature in style_features]
    
    # Define optimizer
    optimizer = optim.Adam([target_img], lr=0.003)
    
    # Start style transfer iterations
    for step in range(iterations):
        target_features = vgg(target_img)
        content_loss = calculate_content_loss(target_features, content_features)
        style_loss = sum([calculate_style_loss(t, s) for t, s in zip(target_features, style_grams)])
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Save and print progress every 100 steps
        if step % 100 == 0:
            print(f'Step {step}, Total Loss: {total_loss.item()}')
            save_image(target_img, output_image_path)  # Save the output image at every step
            
    # Save the final styled image
    save_image(target_img, output_image_path)


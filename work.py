from cap_submission import *
name="./cap-vlm-asajid2"
model, processor, tokenizer = load_trained_model("./" + name)

print(torch.cuda.is_available())
inf1="./" + "cats-hissing-660x287.jpg"
print(f"inference 1 is: {inference(inf1,model,processor,tokenizer)}")
print()

inf2="./" + "plating-magic-for-kids1" + ".jpg"
print(f"inference 2 is: {inference(inf2,model,processor,tokenizer)}")
print()

inf3="./" + "yuji" + ".jpg"
print(f"inference 3 is: {inference(inf3,model,processor,tokenizer)}")
print()

inf4="./" + "sky" + ".jpg"
print(f"inference 4 is: {inference(inf4,model,processor,tokenizer)}")
print()

# style_transfer(
#     content_image_path="./yuji.jpg", # Path to content image
#     style_image_path="./go.jpg",          # Path to style image
#     output_image_path="./styled_image.jpg",                 # Path to save styled image
# )


# Example usage of the style transfer function
style_transfer(
    content_image_path="./yuji.jpg",  # Path to content image
    style_image_path="./sky.jpg",            # Path to style image
    output_image_path="./styled_image.jpg",                 # Path to save styled image
    iterations=1000,                                        # Increased number of iterations
    content_weight=1e3,                                     # Balance between content and style
    style_weight=1e20                                       # Heavier emphasis on style
)
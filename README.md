# Fast Style Transfer 🏎💨🖌️🎨🧠

## Background 📖📕
In this repository, I will do a PyTorch implemention of the fast neural style transfer algorithm described in the paper [Perceptual Losses for Real-Time Style Transfer
and Super-Resolution](https://cs.stanford.edu/people/jcjohns/eccv16/) by Justin Johnson, Alexandre Alahi, and Li Fei-Fei.

This method essentially involves training a model to **approximate** the [optimization based neural style transfer](https://github.com/igreat/artistic-style-net). The benefit is that it runs about 3 orders of magnitude faster!

Because of it's improved inference time, it's feasible to run style transfer in real time as you'll also see in this repository.

<div align="center">
    <img src="images/style_images/starry-night.jpg" alt="Starry Night" width="256"/>
    <img src="images/content_images/bahla-fort.jpg" alt="Bahla Fort" width=256/>
    <img src="images/generated_images/bahla-fort-starry-night.png" alt="Starry Grand Mosque" width="512"/>
</div>

## Results 😎

<!-- original images -->
<p align="center">
    <img src="images/content_images/fruits.jpg" alt="Fruits" width=256/>
    <img src="images/content_images/bahla-fort.jpg" alt="Bahla Fort" width=256/>


<!-- starry night -->
<p align="center">
    <img src="images/style_images/starry-night.jpg" width="200" title="starry-night">
    <img src="images/generated_images/fruits-starry-night.png" width="200" title="produced 2">
    <img src="images/generated_images/bahla-fort-starry-night.png" width="200" title="produced 1">
</p>

<!-- rain princess -->
<p align="center">
    <img src="images/style_images/rain-princess.jpg" width="200" title="style 2">
    <img src="images/generated_images/fruits-rain-princess.png" width="200" title="produced 3">
    <img src="images/generated_images/bahla-fort-rain-princess.png" width="200" title="produced 4">
</p>

## Control and Tradeoffs
Training the model involves a bunch of hyperparameters which include:
- Style weight
- Content weight
- TV regularization to improve smoothness

I've found that when leaving the content weight as 1e2, a style weight ranging from 1e7 or 1e8 works well, but it depends on the style image.

Though I've kept the total variation regulizer, I found that leaving it at 0 yields consistently better results. However, perhaps if its sufficiently small it can be good (the lua implementation from the original paper had it).

## Video Stylization

## Comparison with Optimization Based Method

## Setting This Up On Your Computer 

## How To Style Your Image Or Video

## How To Train Your Own Model



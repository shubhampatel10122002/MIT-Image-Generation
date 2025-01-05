import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import random
import os
import zipfile
from PIL import Image
import matplotlib.pyplot as plt

class FeatureExtractor(nn.Module):
    def __init__(self, channels=1):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.features(x)

class CatalystSegmentation(nn.Module):
    def __init__(self, in_channels=1):
        super(CatalystSegmentation, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        mask = self.decoder(features)
        return mask, features

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Softmax(dim=-1)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        attention = self.attention(x)
        return x + attention * self.conv_block(x)

class EnhancedGenerator(nn.Module):
    def __init__(self, channels=1):
        super(EnhancedGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        self.transformer = nn.Sequential(
            AttentionBlock(256),
            AttentionBlock(256),
            AttentionBlock(256)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, channels, 7, 1, 3),
            nn.Tanh()
        )
        self.segmentation = CatalystSegmentation()

    def forward(self, x):
        x = torch.clamp(x, -1, 1)
        x = torch.nan_to_num(x, 0.0)
        features = self.encoder(x)
        transformed = self.transformer(features)
        output = self.decoder(transformed)
        output = torch.clamp(output + 0.1 * x, -1, 1)
        mask, seg_features = self.segmentation(x)
        return output, mask, seg_features
class CatalystDiscriminator(nn.Module):
    def __init__(self, channels=1):
        super(CatalystDiscriminator, self).__init__()
        
        # Main discriminator path
        self.initial = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        
        self.attention = AttentionBlock(256)
        
        self.final = nn.Sequential(
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
        
        # Catalyst detection branch
        self.catalyst_detector = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Initial layers
        x = self.initial(x)
        x = self.middle(x)
        
        # Store features after middle layers for catalyst detection
        features = x
        
        # Continue with main discriminator path
        x = self.attention(x)
        x = self.final(x)
        
        # Generate catalyst mask from stored features
        catalyst_mask = self.catalyst_detector(features)
        
        return x, catalyst_mask

class PerceptualNetwork(nn.Module):
    def __init__(self):
        super(PerceptualNetwork, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.blocks = nn.ModuleList([
            vgg.features[:4],
            vgg.features[4:9],
            vgg.features[9:16]
        ])
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # Convert grayscale to 3 channels
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

class CatalystRemovalLoss(nn.Module):
    def __init__(self):
        super(CatalystRemovalLoss, self).__init__()

    def forward(self, generated_image, catalyst_mask):
        bright_regions = torch.where(generated_image > 0.8, 1.0, 0.0)
        return torch.mean(bright_regions * catalyst_mask)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.perceptual_net = PerceptualNetwork().cuda() if torch.cuda.is_available() else PerceptualNetwork()

    def forward(self, real, fake):
        real_features = self.perceptual_net(real)
        fake_features = self.perceptual_net(fake)
        loss = 0
        for rf, ff in zip(real_features, fake_features):
            loss += F.mse_loss(ff, rf)
        return loss

def train_enhanced_cycada(zip_path, drive_path, num_epochs=200, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize networks
    G_c2b = EnhancedGenerator().to(device)
    G_b2c = EnhancedGenerator().to(device)
    D_catalyst = CatalystDiscriminator().to(device)
    D_bare = CatalystDiscriminator().to(device)
    
    # Initialize loss functions
    perceptual_loss = PerceptualLoss().to(device)
    catalyst_removal_loss = CatalystRemovalLoss().to(device)
    
    # Create dataset and dataloader
    dataset = CarbonCatalystDataset(load_images_from_zip(zip_path))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizers
    optimizer_G = torch.optim.Adam(
        list(G_c2b.parameters()) + list(G_b2c.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        list(D_catalyst.parameters()) + list(D_bare.parameters()),
        lr=0.0001, betas=(0.5, 0.999)
    )
    
    for epoch in range(num_epochs):
        for i, real_catalyst in enumerate(dataloader):
            real_catalyst = real_catalyst.to(device)
            
            # Generate translations
            fake_bare, fake_mask, _ = G_c2b(real_catalyst)
            recovered_catalyst, rec_mask, _ = G_b2c(fake_bare)
            
            # Calculate losses
            cycle_loss = F.l1_loss(recovered_catalyst, real_catalyst)
            p_loss = perceptual_loss(real_catalyst, recovered_catalyst)
            removal_loss = catalyst_removal_loss(fake_bare, fake_mask)
            
            # Adversarial losses
            d_fake_bare, fake_bare_mask = D_bare(fake_bare)
            d_rec_catalyst, rec_catalyst_mask = D_catalyst(recovered_catalyst)
            
            loss_G_adv = (
                F.mse_loss(d_fake_bare, torch.ones_like(d_fake_bare)) +
                F.mse_loss(d_rec_catalyst, torch.ones_like(d_rec_catalyst))
            )
            
            # Combined generator loss
            loss_G = (
                8.0 * cycle_loss +
                5.0 * p_loss +
                20.0 * removal_loss +
                1.0 * loss_G_adv
            )
            
            # Update generators
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
            # Update discriminators
            optimizer_D.zero_grad()
            d_real_catalyst, real_catalyst_mask = D_catalyst(real_catalyst)
            
            loss_D = (
                F.mse_loss(d_real_catalyst, torch.ones_like(d_real_catalyst)) +
                F.mse_loss(d_rec_catalyst.detach(), torch.zeros_like(d_rec_catalyst)) +
                F.mse_loss(d_fake_bare.detach(), torch.zeros_like(d_fake_bare))
            ) * 0.5
            
            loss_D.backward()
            optimizer_D.step()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}]')
                print(f'Losses - G: {loss_G.item():.4f}, D: {loss_D.item():.4f}')
                print(f'Components - Cycle: {cycle_loss.item():.4f}, '
                      f'Perceptual: {p_loss.item():.4f}, Removal: {removal_loss.item():.4f}')
        
        if (epoch + 1) % 5 == 0:
            save_and_visualize_results(epoch, real_catalyst, G_c2b, G_b2c, device, drive_path)
            save_models(epoch, G_c2b, G_b2c, drive_path)

    return G_c2b, G_b2c
    
  # Dataset and visualization functions
def load_images_from_zip(zip_path, image_size=256):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.crop((0, 0, img.size[0], int(img.size[1] * 0.9)))),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Lambda(lambda x: torch.clamp(x, -1, 1))
    ])

    images = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for filename in zip_ref.namelist():
            if filename.lower().endswith(('.tiff', '.tif')):
                with zip_ref.open(filename) as file:
                    img = Image.open(file).convert('L')
                    if len(images) == 0:  # Preview first image
                        plt.figure(figsize=(10, 5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(img, cmap='gray')
                        plt.title('Original Image')
                        
                        cropped = img.crop((0, 0, img.size[0], int(img.size[1] * 0.9)))
                        plt.subplot(1, 2, 2)
                        plt.imshow(cropped, cmap='gray')
                        plt.title('Cropped Image')
                        plt.show()
                    
                    img_tensor = transform(img)
                    images.append(img_tensor)

    return torch.stack(images)
class CarbonCatalystDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

def save_and_visualize_results(epoch, sample_catalyst, G_c2b, G_b2c, device, drive_path):
    os.makedirs(drive_path, exist_ok=True)

    with torch.no_grad():
        fake_bare, fake_mask, _ = G_c2b(sample_catalyst)
        recovered_catalyst, rec_mask, _ = G_b2c(fake_bare)

        def denorm(x):
            return (x + 1) / 2

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original and reconstructions
        axes[0,0].imshow(denorm(sample_catalyst[0,0]).cpu().numpy(), cmap='gray')
        axes[0,0].set_title('Original (with catalyst)')
        axes[0,0].axis('off')

        axes[0,1].imshow(denorm(fake_bare[0,0]).cpu().numpy(), cmap='gray')
        axes[0,1].set_title('Predicted (without catalyst)')
        axes[0,1].axis('off')

        axes[0,2].imshow(denorm(recovered_catalyst[0,0]).cpu().numpy(), cmap='gray')
        axes[0,2].set_title('Recovered (with catalyst)')
        axes[0,2].axis('off')

        # Segmentation masks
        axes[1,0].imshow(fake_mask[0,0].cpu().numpy(), cmap='hot')
        axes[1,0].set_title('Catalyst Mask (Original)')
        axes[1,0].axis('off')

        axes[1,1].imshow(rec_mask[0,0].cpu().numpy(), cmap='hot')
        axes[1,1].set_title('Catalyst Mask (Recovered)')
        axes[1,1].axis('off')

        # Difference map
        diff = torch.abs(sample_catalyst - recovered_catalyst)
        axes[1,2].imshow(denorm(diff[0,0]).cpu().numpy(), cmap='hot')
        axes[1,2].set_title('Difference Map')
        axes[1,2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(drive_path, f'transformation_epoch_{epoch+1}.png')
        plt.savefig(save_path)
        plt.close()

def save_models(epoch, G_c2b, G_b2c, drive_path):
    models_path = os.path.join(drive_path, 'models')
    os.makedirs(models_path, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'G_c2b_state_dict': G_c2b.state_dict(),
        'G_b2c_state_dict': G_b2c.state_dict(),
    }, os.path.join(models_path, f'generators_epoch_{epoch+1}.pth'))

def preview_random_images(images, num_samples=3):
    plt.figure(figsize=(15, 5))
    indices = random.sample(range(len(images)), min(num_samples, len(images)))

    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[idx][0].numpy(), cmap='gray')
        plt.title(f'Image {idx}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set your paths here
    ZIP_PATH = '/content/drive/MyDrive/Harikrishna_Images/jsn_zip.zip'
    DRIVE_PATH = '/content/drive/MyDrive/Harikrishna_Images/carbon_catalyst_results_4' 
    
    # Mount Google Drive if using Colab
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Preview data before training
    print("Loading and previewing data...")
    images = load_images_from_zip(ZIP_PATH)
    print(f"Loaded {len(images)} images")
    preview_random_images(images)
    
    # Create dataset and dataloader
    dataset = CarbonCatalystDataset(images)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Start training
    print("Starting training...")
    G_c2b, G_b2c = train_enhanced_cycada(ZIP_PATH, DRIVE_PATH)
    
    print("Training completed!")

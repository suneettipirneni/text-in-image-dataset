from imagen_pytorch import Unet, Imagen, ImagenTrainer, t5
from imagen_pytorch.data import Dataset, DataLoader
from dataset import ImgTextDataset 
import torchvision.transforms as T
import torch

unet1 = Unet(
    dim = 128,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True)
)

unet2 = Unet(
    dim = 128,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)


imagen = Imagen(
    unets = unet1,
    image_sizes = 64,
    timesteps = 1000,
    channels = 1,
    cond_drop_prob = 0.1
)

trainer = ImagenTrainer(
    imagen = imagen,
    dl_tuple_output_keywords_names = ('images', 'texts'),
    split_valid_from_train = True
).cuda()

full_dataset = ImgTextDataset(transform=T.Compose([T.Resize(64), T.ToTensor()]))

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.1, 0.1])

def custom_collate_fn(batch):
    images, captions = zip(*batch)
    captions_list = list(captions)
    return torch.utils.data.dataloader.default_collate(images), captions_list

train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=4, collate_fn=custom_collate_fn)

trainer.add_train_dataloader(train_dataloader)
trainer.add_valid_dataloader(val_dataloader)

trainer.load('./checkpoint-30000.pt')

print('outer')
for i in range(200000):
    print('start loop')
    loss = trainer.train_step(unet_number = 1, max_batch_size = 4)
    print(f'unet {i} loss: {loss}')

    if not (i % 100):
        valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 4)
        print(f'valid loss: {valid_loss}')
        trainer.save(f'./checkpoint-{i}.pt')


    if not (i % 25) and trainer.is_main:
        images = trainer.sample(batch_size = 1, texts = ["the text 'auditor'"], return_pil_images = True, cond_scale = 5.) # returns List[Image]
        images[0].save(f'./sample-{i // 100}.png')

    print('end loop')

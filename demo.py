import torch

from uio2.model import UnifiedIOModel
from uio2.preprocessing import UnifiedIOPreprocessor
from uio2.runner import TaskRunner, extract_keypoints, extract_individual_keypoints
from uio2.data_utils import resize_and_pad_default, values_to_tokens

import pylab as plt
from PIL import Image
from tqdm import tqdm
import os

import numpy as np

def detect_keypoints(runner, img_filename):
    #p = runner.prompt.random_prompt("VQA_short_prompt")
    #p = p.replace("{}", "Where are corners of the towel in this image. Return as keypoints.")
    #p = p.replace("{}", "List coordinates of all visible towel corners. Return as keypoints.")
    p = "List coordinates of all visible towel corners in <image_input>"
    example = runner.uio2_preprocessor(text_inputs=p, image_inputs=img_filename, target_modality="text")
    text = runner.predict_text(example, max_tokens=32, detokenize=True, logits_processor=None)
    kps = extract_individual_keypoints(text, example["/meta/image_info"])
    
    return kps, text



def centers_to_tokens(gt_centers, img_shape):
    
    gt_centers_tokens_x = values_to_tokens(gt_centers[:,0] / img_shape[1])
    gt_centers_tokens_y = values_to_tokens(gt_centers[:,1] / img_shape[0])


    gt_centers_tokens_x = [x.decode("utf-8") for x in gt_centers_tokens_x.numpy()]
    gt_centers_tokens_y = [y.decode("utf-8") for y in gt_centers_tokens_y.numpy()]

    gt_centers_text = " , ".join([f"{x} {y}" for x,y in zip(gt_centers_tokens_x, gt_centers_tokens_y)])

    return gt_centers_text


def detect_towel_corners_in_context(runner, train_img_filenames, img_filename):

    from datasets import ClothDataset

    from torchvision.transforms import InterpolationMode
    transform_img_keys = ['image', 'instance', 'label', 'ignore', 'orientation', 'mask']
    transform_tensors = [torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor, torch.ByteTensor]
    transform_img_interpolate = [InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST,  InterpolationMode.NEAREST,  InterpolationMode.BILINEAR, InterpolationMode.NEAREST]

    USE_SEGMENTATION = False
    USE_DEPTH = False

    if USE_SEGMENTATION:
        transform_img_keys.append('segmentation_mask')
        transform_tensors.append(torch.FloatTensor)
        transform_img_interpolate.append(InterpolationMode.NEAREST)

    if USE_DEPTH:
        transform_img_keys.append('depth')
        transform_tensors.append(torch.FloatTensor)
        transform_img_interpolate.append(InterpolationMode.NEAREST)


    transform = [
        # for training without augmentation (same as testing)
        {
            'name': 'ToTensor',
            'opts': {
                'keys': transform_img_keys,
                'type': transform_tensors,
            },			
        },
    ]

    subfolders = [dict(folder='bg=white_desk', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
                    dict(folder='bg=green_checkered', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
                    dict(folder='bg=poster', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
                    dict(folder='bg=festive_tablecloth', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes'])
    ]

    #subfolders = [dict(folder='../demo_cube', annotation='annotations.json', data_subfolders=['.']),]
    #subfolders = [dict(folder='../IJS-examples',data_subfolders=['.'])]

    db = ClothDataset(root_dir='/storage/datasets/ClothDataset/ClothDatasetVICOS/', resize_factor=1, transform_only_valid_centers=1.0, transform=transform, segment_cloth=USE_SEGMENTATION, use_depth=USE_DEPTH, correct_depth_rotation=False, subfolders=subfolders)

    # prepare training data
    train_imgs = []
    train_prompts = []

    for i,img_name in enumerate(train_img_filenames):
        id = db.image_list.index(img_name)        
        sample = db[id]

        img = np.array(sample['image'])
        center = sample['center']

        gt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :]
        gt_centers = gt_centers[:1,:] # only one
        
        gt_centers_text = centers_to_tokens(gt_centers, img.shape[1:])

        train_prompts.append(f"List of all visible corners (X,Y) of the towel in <image_history_{i+1}> is: {gt_centers_text}")
        train_imgs.append(img)

    # Create final prompt and train image
    in_context_prompt = " . ".join(train_prompts)

    final_promp = f"{in_context_prompt}. Find all visible corners of the towel in <image_input>"

    train_imgs = np.stack(train_imgs, axis=0)
    train_imgs = torch.from_numpy(train_imgs)

    batch = runner.uio2_preprocessor(text_inputs=final_promp, image_inputs=img_filename, image_history=train_img_filenames, encode_frame_as_image=None, use_video_audio=False, target_modality="text")
    text = runner.predict_text(batch, max_tokens=128)

    print("response: ",text)

    prompt_sample = db[db.image_list.index(img_filename)]
    img = np.array(prompt_sample['image'])
    center = prompt_sample['center']
    prompt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :] 
    prompt_centers

    print("gt:", centers_to_tokens(prompt_centers, img))

    return text


def eval_blink_examples(runner, subtask):
    from datasets import load_dataset

    data = load_dataset('BLINK-Benchmark/BLINK', subtask)
    
    is_correct_total = 0
    total = 0
    db = tqdm(data['val'], desc=subtask)
    for row in db:
        img = row['image_1']
        prompt = row['prompt']

        anwser = runner.vqa(img, prompt)

        is_correct = row['answer'] in anwser

        #plt.figure()
        #plt.imshow(img)
        
        #print(prompt)
        #print("GROUNDTRUTH: ", row['answer'])
        #print("RESPONSE: ", anwser, (" - CORRECT" if is_correct else " - WRONG"))
        
        #plt.show(block=False)
        #plt.waitforbuttonpress(0.1)
        is_correct_total += is_correct
        total += 1
        is_correct_str = "CORRECT" if is_correct else "WRONG"        
        desc = f"{anwser } - {is_correct_str} - {int(is_correct_total/total*100)}%"
        db.set_postfix_str(desc)
        
    
    print(f"{subtask}: total correct {is_correct_total}/{len(data['val'])}, accuracy={is_correct_total/len(data['val'])}")

    return is_correct_total, len(data['val'])


if __name__ == "__main__":

    dev = torch.device("cuda:0")

    preprocessor = UnifiedIOPreprocessor.from_pretrained("allenai/uio2-preprocessor", tokenizer="/home/domen/Projects/vision-language/llama2/tokenizer.model")

    #model = UnifiedIOModel.from_pretrained("allenai/uio2-xxl")
    model = UnifiedIOModel.from_pretrained("allenai/uio2-large")
    
    state = torch.load("./exp/vicos-towels/model=allenai/uio2-large_resize_factor=1_batchsize=24/num_train_epoch=10/depth=False/checkpoint.pth")
    model_state_dict = {k.replace("module.",""):v for k,v in state['model_state_dict'].items()}
    model.load_state_dict(model_state_dict, strict=True)
    model.to(dev)

    runner = TaskRunner(model, preprocessor)

    if True:
        # eval_blink_examples(runner, "Relative_Depth")
        # eval_blink_examples(runner, "Relative_Depth")
        # eval_blink_examples(runner, "Spatial_Relation")
        # eval_blink_examples(runner, "Spatial_Relation")
        # eval_blink_examples(runner, "Relative_Reflectance")
        # eval_blink_examples(runner, "Relative_Reflectance")
        # eval_blink_examples(runner, "IQ_Test")
        # eval_blink_examples(runner, "IQ_Test")
        # eval_blink_examples(runner, "Counting")
        # eval_blink_examples(runner, "Counting")
        # eval_blink_examples(runner, "Object_Localization")
        # eval_blink_examples(runner, "Object_Localization")
        

        img_filename = "/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view0_ls4_camera0.jpg"
        #img_filename = "/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view19_ls1_camera0.jpg"

        #detect_towel_corners_in_context(runner, ["/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view0_ls3_camera0.jpg",
        #                                         "/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view0_ls5_camera0.jpg",
        #                                         "/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view0_ls6_camera0.jpg",
        #                                         "/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=white_desk/cloth=big_towel/rgb/image_0000_view0_ls0_camera0.jpg"], img_filename)
        
        kps, text = detect_keypoints(runner, img_filename)
        print(text)
        print(kps)

        import PIL
        plt.figure()
        plt.imshow(PIL.Image.open(img_filename))
        plt.plot(kps[:,0],kps[:,1],'.')
        plt.show(block=True)

        #box = runner.refexp(img_filename, "towel")
        #print(box)

        #seg= runner.segmentation_box(img_filename, "towel", [578.4984984984985, 243.0630630630631, 1401.081081081081, 981.081081081081])

        #caption = runner.image_captioning(img_filename)
        #print(caption)

        #plt.figure()
        #plt.imshow(seg[0])
        #
        #image = runner.image_generation("a cat")
        #plt.show(block=True)
    
    if False:
        folders = ["/storage/group/RTFM/competition2024/demo_examples/competition_output_run1",
                "/storage/group/RTFM/competition2024/demo_examples/competition_output_run2"]


        for f in folders:
            for img_filename in tqdm(os.listdir(f)):

                if not img_filename.endswith('input.png'):
                    continue

                seg = runner.segmentation_class(os.path.join(f,img_filename),"towel")

                for i,s in enumerate(seg):
                    img = Image.fromarray(s)
                    img.save(os.path.join(f,img_filename.replace('input.png',f"output_unified_io2_instance_{i}.png")))



from mmdet.apis import init_detector, inference_detector, show_result_pyplot,show_result
import mmcv
import os
config_file = '../configs/htc/htc_without_semantic_r50_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../work_dirs/htc_without_semantic_r50_fpn_1x/epoch_11.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# img = 'D:\\dataset\\f_track\\fish_super_resolution\\fish_test\\test_left\\01_004172866828_12625.png'
# result = inference_detector(model, img)
# result = (result[0], result[1][0])
# show_result_pyplot(img, result, model.CLASSES)

i = '01_000000589598_05298.png'
filePath = 'D:\\dataset\\f_track\\fish_test\\fish_image'
# file_name = os.listdir(filePath)
# save_path = 'D:\\program\\instance_segmentation\\mmdetection_scoring\\results\\gt_117\\'

    # ret_val, img = cam.read()
img = filePath + '\\' + i
print(img)
result = inference_detector(model, img)
result = (result[0], result[1][0])
# save_name = save_path + i
    # show_result_pyplot(img, result, model.CLASSES, out_file=None)
show_result(img, result, model.CLASSES)


# video = mmcv.VideoReader('D:\\dataset\\f_track\\gt_107.flv')
# for frame in video:
#     result = inference_detector(model, frame)
#     # print(result[0])
#     # print(result[1])
#     # print(result[1][0])
#     result = (result[0], result[1][0])
#     show_result(frame, result, model.CLASSES, wait_time=1)
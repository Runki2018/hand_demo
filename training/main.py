from datasets.ZHHand_crop_img import ZHHandDataSet
from training.train_ZHhand import ZHHandTrain

if __name__ == '__main__':
    test_file = 'first_subset/test.json'
    train_file = 'first_subset/train.json'
    ds_test = ZHHandDataSet(ann_file=test_file)
    ds_train = ZHHandDataSet(ann_file=train_file)
    # you should manually change the setting of model loading in the file 'base_model' on line 162
    train_PoseModel = ZHHandTrain(exp_name='train_HRNet_1',
                                  model_c=48,   # HRNet 32 or 48/ PoseResNet 50
                                  ds_val=ds_test,
                                  ds_train=ds_train)
    train_PoseModel.run()

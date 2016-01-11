##Run preprocessing
python prepro.py --input_json coco/coco_raw.json --num_val 50 --num_test 50 --num_train 100 --image_feats_Tr /home/angeliki/sata/DATA/MSCOCO/features/train2014/features_fc7.mat --image_feats_Ts /home/angeliki/sata/DATA/MSCOCO/features/val2014/features_fc7.mat --image_list_Tr /home/angeliki/sata/DATA/MSCOCO/features/train2014/features_fc7.index --image_list_Ts /home/angeliki/sata/DATA/MSCOCO/features/val2014/features_fc7.index

##Run Code
th train.lua -input_json coco/data_with_feats_40000.json -input_h5 coco/data_with_feats_40000.h5 -gpuid 0 -backend nn

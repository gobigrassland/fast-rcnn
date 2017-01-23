#include "caffe/util/parse_config.hpp"
#include "caffe/3rdparty/cfgparser.h"
#include <iostream>
#include <glog/logging.h>

ParseConfig::ParseConfig(const string& cfg_file)
{
	this->cfg_file = cfg_file;
}


ParseConfig::~ParseConfig()
{
}

void ParseConfig::InitializeTrainConfig()
{
	TRAIN_CFG.SCALES.resize(1);
	TRAIN_CFG.SCALES[0] = 600;
	TRAIN_CFG.MAX_SIZE = 1000;
	TRAIN_CFG.IMS_PER_BATCH = 2;
	TRAIN_CFG.BATCH_SIZE = 128;
	TRAIN_CFG.FG_FRACTION = 0.25;
	TRAIN_CFG.FG_THRESH = 0.5;
	TRAIN_CFG.BG_THRESH_HI = 0.5;
	TRAIN_CFG.BG_THRESH_LO = 0.1;
	TRAIN_CFG.USE_FLIPPED = true;
	TRAIN_CFG.BBOX_REG = true;
	TRAIN_CFG.BBOX_THRESH = 0.5;
	TRAIN_CFG.SNAPSHOT_ITERS = 10000;
	TRAIN_CFG.SNAPSHOT_INFIX = "";
	TRAIN_CFG.USE_PREFETCH = false;
}

void ParseConfig::InitializeTestConfig()
{
	TEST_CFG.SCALES.resize(1);
	TEST_CFG.SCALES[0] = 600;
	TEST_CFG.MAX_SIZE = 1000;
	TEST_CFG.NMS = 0.3;
	TEST_CFG.SVM = false;
	TEST_CFG.BBOX_REG = true;
}

void ParseConfig::InitializeDeployConfig()
{
    DEPLOY_CFG.SCALES.resize(1);
    DEPLOY_CFG.SCALES[0] = 600;
    DEPLOY_CFG.MAX_SIZE = 1000;
    DEPLOY_CFG.NMS = 0.3;
    DEPLOY_CFG.CONF_THRESH = 0.8;
}

void ParseConfig::InitializeCommonConfig()
{
	COMMON_CFG.DEDUP_BOXES = 0.0625;
	COMMON_CFG.PIXEL_MEANS.resize(3);
	COMMON_CFG.PIXEL_MEANS[0] = 102.9801;
	COMMON_CFG.PIXEL_MEANS[1] = 115.9465;
	COMMON_CFG.PIXEL_MEANS[2] = 122.7717;
	COMMON_CFG.RNG_SEED = 3;
}

void ParseConfig::ParseTrainConfig()
{
	ConfigParser_t cfg;
        CHECK(cfg.readFile(cfg_file) == 0) << "Cannot open config file";
	CHECK(cfg.getValue("TRAIN", "SCALES", &TRAIN_CFG.SCALES));
	CHECK(cfg.getValue("TRAIN", "MAX_SIZE", &TRAIN_CFG.MAX_SIZE));
	CHECK(cfg.getValue("TRAIN", "IMS_PER_BATCH", &TRAIN_CFG.IMS_PER_BATCH));
	CHECK(cfg.getValue("TRAIN", "BATCH_SIZE", &TRAIN_CFG.BATCH_SIZE));
	CHECK(cfg.getValue("TRAIN", "FG_FRACTION", &TRAIN_CFG.FG_FRACTION));
	CHECK(cfg.getValue("TRAIN", "FG_THRESH", &TRAIN_CFG.FG_THRESH));
	CHECK(cfg.getValue("TRAIN", "BG_THRESH_HI", &TRAIN_CFG.BG_THRESH_HI));
	CHECK(cfg.getValue("TRAIN", "BG_THRESH_LO", &TRAIN_CFG.BG_THRESH_LO));
	CHECK(cfg.getValue("TRAIN", "USE_FLIPPED", &TRAIN_CFG.USE_FLIPPED));
	CHECK(cfg.getValue("TRAIN", "BBOX_REG", &TRAIN_CFG.BBOX_REG));
	CHECK(cfg.getValue("TRAIN", "BBOX_THRESH", &TRAIN_CFG.BBOX_THRESH));
	CHECK(cfg.getValue("TRAIN", "SNAPSHOT_ITERS", &TRAIN_CFG.SNAPSHOT_ITERS));
	CHECK(cfg.getValue("TRAIN", "SNAPSHOT_INFIX", &TRAIN_CFG.SNAPSHOT_INFIX));
	CHECK(cfg.getValue("TRAIN", "USE_PREFETCH", &TRAIN_CFG.USE_PREFETCH));
}


void ParseConfig::ParseTestConfig()
{
	ConfigParser_t cfg;
        CHECK(cfg.readFile(cfg_file) == 0) << "Cannot open config file";
	CHECK(cfg.getValue("TEST", "SCALES", &TEST_CFG.SCALES));
	CHECK(cfg.getValue("TEST", "MAX_SIZE", &TEST_CFG.MAX_SIZE));
	CHECK(cfg.getValue("TEST", "NMS", &TEST_CFG.NMS));
	CHECK(cfg.getValue("TEST", "SVM", &TEST_CFG.SVM));
	CHECK(cfg.getValue("TEST", "BBOX_REG", &TEST_CFG.BBOX_REG));
}

void ParseConfig::ParseDeployConfig()
{
        ConfigParser_t cfg;
        CHECK(cfg.readFile(cfg_file) == 0) << "Cannot open config file";
	CHECK(cfg.getValue("DEPLOY", "SCALES", &DEPLOY_CFG.SCALES));
	CHECK(cfg.getValue("DEPLOY", "MAX_SIZE", &DEPLOY_CFG.MAX_SIZE));
	CHECK(cfg.getValue("DEPLOY", "NMS", &DEPLOY_CFG.NMS));
	CHECK(cfg.getValue("DEPLOY", "CONF_THRESH", &DEPLOY_CFG.CONF_THRESH));    
}

void ParseConfig::ParseCommonConfig()
{
	ConfigParser_t cfg;
        CHECK(cfg.readFile(cfg_file) == 0) << "Cannot open config file";	
	CHECK(cfg.getValue("COMMON", "DEDUP_BOXES", &COMMON_CFG.DEDUP_BOXES));
	CHECK(cfg.getValue("COMMON", "PIXEL_MEANS", &COMMON_CFG.PIXEL_MEANS));
	CHECK(cfg.getValue("COMMON", "RNG_SEED", &COMMON_CFG.RNG_SEED));
	CHECK(cfg.getValue("COMMON", "ROOT_DIR", &COMMON_CFG.ROOT_DIR));
	CHECK(cfg.getValue("COMMON", "EXP_DIR", &COMMON_CFG.EXP_DIR));
	CHECK(cfg.getValue("COMMON", "IMGS_LIST", &COMMON_CFG.IMGS_LIST));
	CHECK(cfg.getValue("COMMON", "CLASSES_LIST", &COMMON_CFG.CLASSES_LIST));
	CHECK(cfg.getValue("COMMON", "SS_MAT", &COMMON_CFG.SS_MAT));
	CHECK(cfg.getValue("COMMON", "DIR_IMGS", &COMMON_CFG.DIR_IMGS));
	CHECK(cfg.getValue("COMMON", "DIR_ANNOTATIONS", &COMMON_CFG.DIR_ANNOTATIONS));
}
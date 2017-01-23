#ifndef _PARSE_CONFIG_HPP_
#define _PARSE_CONFIG_HPP_

#include <string>
#include <vector>
using namespace std;

struct TRAIN
{
	vector<int> SCALES;
	int MAX_SIZE;
	int IMS_PER_BATCH;
	int BATCH_SIZE;
	float FG_FRACTION;
	float FG_THRESH;
	float BG_THRESH_HI;
	float BG_THRESH_LO;
	bool USE_FLIPPED;
	bool BBOX_REG;
	float BBOX_THRESH;
	int SNAPSHOT_ITERS;
	string SNAPSHOT_INFIX;
	bool USE_PREFETCH;
};

struct TEST
{
	vector<int> SCALES;
	int MAX_SIZE;
	float NMS;
	bool SVM;
	bool BBOX_REG;
};

struct DEPLOY
{
	vector<int> SCALES;
	int MAX_SIZE;
        float NMS;
        float CONF_THRESH;
};

struct COMMON
{
	float DEDUP_BOXES;
	vector<float> PIXEL_MEANS;
	int RNG_SEED;
	string ROOT_DIR;
	string EXP_DIR;
	string IMGS_LIST;
	string CLASSES_LIST;
	string SS_MAT;
	string DIR_IMGS;
	string DIR_ANNOTATIONS;
};

class ParseConfig
{
	public:
		
	ParseConfig(const string& cfg_file);
	
	~ParseConfig();
	
	void InitializeTrainConfig();
	void ParseTrainConfig();
        struct TRAIN GetTrainConfig() const {return TRAIN_CFG;}
	
	void InitializeTestConfig();
	void ParseTestConfig();
        struct TEST GetTestConfig() const {return TEST_CFG;}
	
	void InitializeDeployConfig();
	void ParseDeployConfig();
        struct DEPLOY GetDeployConfig() const {return DEPLOY_CFG;}
	
	void InitializeCommonConfig();
	void ParseCommonConfig();
        struct COMMON GetCommonConfig() const {return COMMON_CFG;}
	
	string cfg_file;
	struct TRAIN TRAIN_CFG;
	struct TEST TEST_CFG;
	struct DEPLOY DEPLOY_CFG;
	struct COMMON COMMON_CFG;
	
};


#endif
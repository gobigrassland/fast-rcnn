#include "caffe/util/roi_data_extractor.hpp"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <map>
#include "caffe/3rdparty/matio.h"
#include <math.h>

#include <sys/stat.h>
#include <limits>
#include <algorithm>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe
{
    
    ROIDataExtractor::ROIDataExtractor(std::string dir_imgs,
                         std::string path_img_list,
                         std::string path_classes_list,
                         std::string path_annotation,
                         std::string path_selective_search_mat,
                         bool use_flipped)
    {
        dir_imgs_ = dir_imgs;
        path_img_list_ = path_img_list;
        path_classes_list_ = path_classes_list;
        path_annotation_ = path_annotation;
        path_selective_search_mat_ = path_selective_search_mat;
        use_flipped_ = use_flipped;
    }
    ROIDataExtractor::~ROIDataExtractor()
    {

    }
    
    void ROIDataExtractor::getAttribute(pugi::xml_node node,
		std::string name,
		std::vector<std::string> &labels,
		std::vector<std::vector<double> > &bndboxes)
     {
	for(pugi::xml_node subnode = node.first_child(); subnode; subnode = subnode.next_sibling())
	{
		if(strcmp(subnode.name(), name.c_str()) == 0)
		{
			for(pugi::xml_node _node = subnode.first_child(); _node; _node = _node.next_sibling())
			{
				if(strcmp(_node.name(), "name") == 0)
					labels.push_back(_node.first_child().value());
				if(strcmp(_node.name(), "bndbox") == 0)
				{
					double xmin=-1, xmax = -1, ymin = -1, ymax = -1;
					for(pugi::xml_node node_bnd = _node.first_child(); node_bnd; node_bnd = node_bnd.next_sibling())
					{
						double val = atof(node_bnd.first_child().value());
						if(strcmp(node_bnd.name(), "xmin") == 0)
							xmin = val;
						if(strcmp(node_bnd.name(), "ymin") == 0)
							ymin = val;
						if(strcmp(node_bnd.name(), "xmax") == 0)
							xmax = val;
						if(strcmp(node_bnd.name(), "ymax") == 0)
							ymax = val;
					}
					double temp[4];
					temp[0] = xmin - 1;
					temp[1] = ymin - 1;
					temp[2] = xmax - 1;
					temp[3] = ymax - 1;
					bndboxes.push_back(std::vector<double>(temp, temp + 4));
				}
			}
		}
		getAttribute(subnode, name, labels, bndboxes);
	}
    }
    
    void ROIDataExtractor::bbox_overlaps(const std::vector<std::vector<double> > &boxes,
	const std::vector<std::vector<double> > &gt_boxes,
	std::vector<std::vector<double> > &overlaps)
    {
	if (boxes.size() == 0 || gt_boxes.size() == 0)
		return;
	overlaps.resize(boxes.size());
	for (int i = 0; i < boxes.size(); i++)
	{
		double box_area = (boxes[i][2] - boxes[i][0] + 1) * (boxes[i][3] - boxes[i][1] + 1);
		for (int j = 0; j < gt_boxes.size(); j ++)
		{
			double gt_box_area = (gt_boxes[j][2] - gt_boxes[j][0] + 1) * (gt_boxes[j][3] - gt_boxes[j][1] + 1);
			double left = std::max(boxes[i][0], gt_boxes[j][0]), right = std::min(boxes[i][2], gt_boxes[j][2]);
			double top = std::max(boxes[i][1], gt_boxes[j][1]), bottom = std::min(boxes[i][3], gt_boxes[j][3]);
			double dx = std::max(right - left + 1, 0.0);
			double dy = std::max(bottom - top + 1, 0.0);

			double overlap_area = dx * dy;
			double ratio = overlap_area / (box_area + gt_box_area - overlap_area);
			overlaps[i].push_back(ratio);
		}
	}
    }
    
    
    bool ROIDataExtractor::roi_data_extract(std::vector<ROI>& roidb)
    {
        std::vector<std::string> list_imgs;
	std::ifstream infile(path_img_list_.c_str());
	std::string name;
	while(infile >> name)
		list_imgs.push_back(name);

        double EPS = std::numeric_limits<double>::epsilon();
	//map class of images to index
	std::map<std::string, int> class_to_ind;
	std::map<int, std::string> ind_to_class;
	std::ifstream inclasses(path_classes_list_.c_str());
	std::string class_name;
	int ind = 0;
	while(inclasses >> class_name)
	{
		class_to_ind[class_name] = ind;
		ind_to_class[ind] = class_name;
		ind ++;
	}

	
        LOG(INFO) << "Parse xml files";
	std::vector<ROI> gt_roidb(list_imgs.size());
	for(int i = 0; i < list_imgs.size(); i ++)
	{
		pugi::xml_document doc;
		pugi::xml_parse_result result = doc.load_file((path_annotation_ + "/" + list_imgs[i] + ".xml").c_str());
                CHECK(result.status == 0) << std::string(path_annotation_ + "/" + list_imgs[i] + ".xml");
		std::vector<std::string> labels;
		std::vector<std::vector<double> > bndboxes;
		getAttribute(doc.first_child(), "object", labels, bndboxes);
		if (labels.size() != bndboxes.size())
			continue;
		ROI roi;
		roi.boxes = bndboxes;
		std::vector<int> gt_classes;
		for(int j =0 ;j < labels.size(); j ++)
		{
			gt_classes.push_back(class_to_ind[labels[j]]);
		}
		roi.gt_classes = gt_classes;
		for(int j = 0; j < labels.size(); j ++)
		{
			double temp[3];
			temp[0] = (double)j;
			temp[1] = (double)class_to_ind[labels[j]];
			temp[2] = (double)1;
			//roi.gt_overlaps.push_back(std::vector<double>((double)j, (double)class_to_ind[labels[j]], 1));
			roi.gt_overlaps.push_back(std::vector<double>(temp, temp+3));
		}
		roi.flipped = false;

		gt_roidb[i] = roi;
	}

	//read region proposals saved as a mat format file
	mat_t *matfp = Mat_Open(path_selective_search_mat_.c_str(), MAT_ACC_RDONLY);
        CHECK(matfp) << "Error opening the mat file";
        
	matvar_t *mat_boxes = Mat_VarRead(matfp, (char*)"boxes");
        CHECK(mat_boxes) << "Error reading boxes";
        
	unsigned num_cell = 1;
	for (int i = 0; i < mat_boxes->rank; i++)
		num_cell *= mat_boxes->dims[i];

        LOG(INFO) << "Parse object proposals";
	std::vector<ROI> ss_roidb(num_cell);
	for (int i = 0; i < num_cell; i++)
	{
		matvar_t *cell = Mat_VarGetCell(mat_boxes, i);
		unsigned cell_shape[2] = { 0, 0 };
		for (int j = 0; j < cell->rank; j++)
			cell_shape[j] = cell->dims[j];
		const double *cell_data = static_cast<const double*>(cell->data);
		ROI roidb;
		roidb.boxes.resize(cell_shape[0]);
		roidb.gt_classes.resize(cell_shape[0]);
		roidb.flipped = false;
		for (int j = 0; j < cell_shape[0]; j++)
		{
			double temp[4];
			temp[0] = cell_data[j + cell_shape[0]] - 1;
			temp[1] = cell_data[j] - 1;
			temp[2] = cell_data[j + 3 * cell_shape[0]] - 1;
			temp[3] = cell_data[j + 2 * cell_shape[0]] - 1;
			roidb.boxes[j] = std::vector<double>(std::vector<double>(temp, temp+4));
		}
		ss_roidb[i] = roidb;
	}
        CHECK(gt_roidb.size() == ss_roidb.size()) << "Dimensions do not match between ground truth and object proposal";
	
        LOG(INFO) << "Compute overlaps";
        for (int i = 0; i < ss_roidb.size(); i ++)
	{
		std::vector<std::vector<double> > overlaps;
		bbox_overlaps(ss_roidb[i].boxes, gt_roidb[i].boxes, overlaps);
		int num_gt = gt_roidb[i].boxes.size();
		ss_roidb[i].gt_overlaps.resize(overlaps.size());
		for (int j = 0; j < overlaps.size(); j ++)
		{
			int ind = 0;
			double val = overlaps[j][ind];
			for (int k = 1; k < num_gt; k ++)
			{
				if (overlaps[j][k] > val)
				{
					ind = k;
					val = overlaps[j][ind];
				}
			}
			if (val > 0)
			{
				double temp[3];
				temp[0] = (double)ind;
				temp[1] = (double)gt_roidb[i].gt_classes[ind];
				temp[2] = val;
				ss_roidb[i].gt_overlaps[j] = std::vector<double>(temp, temp+3);
			}
			else
			{
				double temp[3];
				temp[0] = (double)ind;
				temp[1] = 0;
				temp[2] = val;
				ss_roidb[i].gt_overlaps[j] = std::vector<double>(temp, temp+3);
			}
			
		}
	}

	for (int i = 0; i < ss_roidb.size(); i ++)
	{
		ss_roidb[i].image = list_imgs[i];
		for (int j = gt_roidb[i].gt_classes.size() - 1; j >= 0;  j--)
		{
			ss_roidb[i].boxes.insert(ss_roidb[i].boxes.begin(), gt_roidb[i].boxes[j]);
			ss_roidb[i].gt_classes.insert(ss_roidb[i].gt_classes.begin(), gt_roidb[i].gt_classes[j]);
			ss_roidb[i].gt_overlaps.insert(ss_roidb[i].gt_overlaps.begin(), gt_roidb[i].gt_overlaps[j]);
		}
	}

        if(use_flipped_)
        {
        	LOG(INFO) << "Appending horizontally-flipped training examples";
            int num_roidb = ss_roidb.size();
            ss_roidb.resize(2*num_roidb);
            for(int i = 0; i < num_roidb; i ++)
            {
            	ss_roidb[i+num_roidb]= ss_roidb[i];
            	ss_roidb[i+num_roidb].flipped = true;
            	cv::Mat img = cv::imread(dir_imgs_ + "/" + ss_roidb[i].image + ".jpg");
                CHECK(img.data) << "Cannot open " << std::string(dir_imgs_ + "/" + ss_roidb[i].image + ".jpg");
            	int height = img.rows;
            	int width = img.cols;
                for(int k = 0; k < ss_roidb[i].boxes.size(); k ++)
                {
                    ss_roidb[i+num_roidb].boxes[k][0] = width - ss_roidb[i+num_roidb].boxes[k][0] - 1;
                    ss_roidb[i+num_roidb].boxes[k][2] = width - ss_roidb[i+num_roidb].boxes[k][2] - 1;
                    ss_roidb[i+num_roidb].boxes[k][1] = height - ss_roidb[i+num_roidb].boxes[k][1] - 1;
                    ss_roidb[i+num_roidb].boxes[k][3] = height - ss_roidb[i+num_roidb].boxes[k][3] - 1;
                }
            }
        }

	//compute regression values of rois for each image
        LOG(INFO) << "Compute regression target";
	for (int i = 0; i < ss_roidb.size(); i ++)
	{
		ss_roidb[i].targets.resize(ss_roidb[i].boxes.size());
		// Indices of ground-truth ROIs
		std::vector<int> gt_inds;
		int ind = 0;
		do 
		{
			gt_inds.push_back(ind++);
		} while ((int)ss_roidb[i].gt_overlaps[ind][2] == 1);

		// Indices of example for which we try to make predictions
		std::vector<int> ex_inds;
                std::vector<int> no_ex_inds;
		for (int ind = 0; ind < ss_roidb[i].gt_overlaps.size(); ind ++)
		{
			if (ss_roidb[i].gt_overlaps[ind][2] >= 0.5)
				ex_inds.push_back(ind);
                        else
                            no_ex_inds.push_back(ind);
		}
		for (int j = 0; j < ex_inds.size(); j ++)
		{
			int ind = ex_inds[j];
			std::vector<double> *ex_box = &(ss_roidb[i].boxes[ind]);
			double ex_width = (*ex_box)[2] - (*ex_box)[0] + EPS;
			double ex_height = (*ex_box)[3] - (*ex_box)[1] + EPS;
			double ex_ctr_x = (*ex_box)[0] + 0.5 * ex_width;
			double ex_ctr_y = (*ex_box)[1] + 0.5 * ex_height;

			int target_ind = ss_roidb[i].gt_overlaps[ind][0];
			std::vector<double> *gt_box = &(ss_roidb[i].boxes[target_ind]);
			double gt_width = (*gt_box)[2] - (*gt_box)[0] + EPS;
			double gt_height = (*gt_box)[3] - (*gt_box)[1] + EPS;
			double gt_ctr_x = (*gt_box)[0] + 0.5 * gt_width;
			double gt_ctr_y = (*gt_box)[1] + 0.5 * gt_height;

			double target_dx = (gt_ctr_x - ex_ctr_x) / ex_width;
			double target_dy = (gt_ctr_y - ex_ctr_y) / ex_height;
			double target_dw = log(gt_width / ex_width);
			double target_dh = log(gt_height / ex_height);
			double temp[5];
			temp[0] = (double)ss_roidb[i].gt_overlaps[ind][1];
			temp[1] = target_dx;
			temp[2] = target_dy;
			temp[3] = target_dw;
			temp[4] = target_dh;
			ss_roidb[i].targets[ind] = std::vector<double>(temp, temp+5);
		}
                for(int j = 0; j < no_ex_inds.size(); j ++)
                {   
                    int ind  = no_ex_inds[j];
                    ss_roidb[i].targets[ind].resize(5);
                }
	}
	//compute mean and std of regression for each categories
	int num_classes = ind_to_class.size();
	std::vector<int> class_counts(num_classes);
	std::vector<std::vector<double> > sums(num_classes), squared_sums(num_classes), means(num_classes), stds(num_classes);
	for (int i = 0; i < num_classes; i ++)
	{
		sums[i].resize(4);
		squared_sums[i].resize(4);
		means[i].resize(4);
		stds[i].resize(4);
	}
        
        LOG(INFO) << "Compute mean and stdev";
	for (int i = 0; i < ss_roidb.size(); i ++)
	{
		for (int j = 0; j < ss_roidb[i].targets.size(); j ++)
		{
			if (ss_roidb[i].targets[j].size() == 0)
				continue;
			std::vector<double> *target_ptr = &(ss_roidb[i].targets[j]);
			int target_cls_ind = (*target_ptr)[0];
			if (target_cls_ind > 0)
			{
				class_counts[target_cls_ind] += 1;
				sums[target_cls_ind][0] += (*target_ptr)[1];
				sums[target_cls_ind][1] += (*target_ptr)[2];
				sums[target_cls_ind][2] += (*target_ptr)[3];
				sums[target_cls_ind][3] += (*target_ptr)[4];

				squared_sums[target_cls_ind][0] += (*target_ptr)[1] * (*target_ptr)[1];
				squared_sums[target_cls_ind][1] += (*target_ptr)[2] * (*target_ptr)[2];
				squared_sums[target_cls_ind][2] += (*target_ptr)[3] * (*target_ptr)[3];
				squared_sums[target_cls_ind][3] += (*target_ptr)[4] * (*target_ptr)[4];
			}
		}
	}
	for (int i = 1; i < num_classes; i ++)
	{
		means[i][0] = sums[i][0] / class_counts[i];
		means[i][1] = sums[i][1] / class_counts[i];
		means[i][2] = sums[i][2] / class_counts[i];
		means[i][3] = sums[i][3] / class_counts[i];

		squared_sums[i][0] /= class_counts[i];
		squared_sums[i][1] /= class_counts[i];
		squared_sums[i][2] /= class_counts[i];
		squared_sums[i][3] /= class_counts[i];
	}

        
	for (int i = 1; i < num_classes; i ++)
	{
		stds[i][0] = sqrt(squared_sums[i][0] - means[i][0] * means[i][0] + EPS);
		stds[i][1] = sqrt(squared_sums[i][1] - means[i][1] * means[i][1] + EPS);
		stds[i][2] = sqrt(squared_sums[i][2] - means[i][2] * means[i][2] + EPS);
		stds[i][3] = sqrt(squared_sums[i][3] - means[i][3] * means[i][3] + EPS);
	}
        struct stat sb;
        if(stat("data/cache", &sb) == 0 && S_ISDIR(sb.st_mode))
            DLOG(INFO) << "Directory of data/cache exists";
        else
        {
            int mkflag = mkdir("data/cache", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            CHECK(mkflag != -1) << "Error creating directory";
        }
        FILE* fid = fopen("data/cache/mean_std.txt", "wb");
        fwrite(&num_classes, sizeof(int), 1, fid);
        for(int i = 0; i < num_classes; i ++)
        {
            fwrite(means[i].data(), sizeof(double), 4, fid);
            fwrite(stds[i].data(), sizeof(double), 4, fid);
        }
        fclose(fid);
	for (int i = 0; i < ss_roidb.size(); i ++)
	{
		std::vector<std::vector<double> > *target = &(ss_roidb[i].targets);
		int num_target = (*target).size();
		for (int j = 0; j < num_target; j ++)
		{
			if ((*target)[j].size() == 0)
				continue;
			int ind_cls_target = int((*target)[j][0]);
			if (ind_cls_target > 0)
			{
				(*target)[j][1] -= means[ind_cls_target][0];
				(*target)[j][1] /= stds[ind_cls_target][0];
				(*target)[j][2] -= means[ind_cls_target][1];
				(*target)[j][2] /= stds[ind_cls_target][1];
				(*target)[j][3] -= means[ind_cls_target][2];
				(*target)[j][3] /= stds[ind_cls_target][2];
				(*target)[j][4] -= means[ind_cls_target][3];
				(*target)[j][4] /= stds[ind_cls_target][3];
			}
		}
	}
    LOG(INFO) << "Number of training examples: " << ss_roidb.size();
        roidb = ss_roidb;
        return true;
    }
}

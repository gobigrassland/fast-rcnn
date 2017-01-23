#ifndef ROI_DATA_EXTRACTOR_HPP
#define ROI_DATA_EXTRACTOR_HPP

#include "caffe/3rdparty/pugixml.hpp"
#include <vector>
#include <string>


namespace caffe
{
    struct ROI
    {
	std::string image;
	std::vector<std::vector<double> > boxes;
	std::vector<int> gt_classes;
	// gt_overlaps[][2]: maximum of IoU overlap between each example ROI and all ground-truth ROIs
	// gt_overlaps[][1]: label of the ground-truth ROI corresponding to the maximum
	// gt_overlaps[][0]: index of the ground-truth ROI
	std::vector<std::vector<double> > gt_overlaps;
	std::vector<std::vector<double> > targets;
	bool flipped;
    };
    
    class ROIDataExtractor
    {
    public:
        ROIDataExtractor(std::string dir_imgs,
                         std::string path_img_list,
                         std::string path_classes_list,
                         std::string path_annotation,
                         std::string path_selective_search_mat,
                         bool use_flipped);
        ~ROIDataExtractor();
        
        void getAttribute(pugi::xml_node node,
		          std::string name,
		          std::vector<std::string> &labels,
		          std::vector<std::vector<double> > &bndboxes);
        
        
        void bbox_overlaps(const std::vector<std::vector<double> > &boxes,
                           const std::vector<std::vector<double> > &gt_boxes,
	                   std::vector<std::vector<double> > &overlaps);
        
        
        bool roi_data_extract(std::vector<ROI>& roidb);
        
    private:
        std::string dir_imgs_;
        std::string path_img_list_;
        std::string path_classes_list_;
        std::string path_annotation_;
        std::string path_selective_search_mat_;
        bool use_flipped_;
        
    };
    
    
}

#endif /* ROI_DATA_EXTRACTOR_HPP */


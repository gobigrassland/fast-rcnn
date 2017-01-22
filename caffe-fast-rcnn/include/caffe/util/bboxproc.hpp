#ifndef BBOXPROC_HPP
#define BBOXPROC_HPP

#include <glog/logging.h>
// non-maximum suppression for cpu
template <typename T>
void nms_cpu(const std::vector<std::vector<T> >& boxes,
	const std::vector<T>& scores,
	std::vector<int>& ind_selected,
	float overlap_thresh)
{
    CHECK(boxes.size() == scores.size());
    CHECK(boxes.size());
	int num_vec = boxes.size();
	std::vector<T> area_boxes(num_vec);
	std::vector<std::pair<T, int> > scores_pair(num_vec);
	for (int i = 0; i < num_vec; i++)
	{
		area_boxes[i] = (boxes[i][2] - boxes[i][0] + 1) * (boxes[i][3] - boxes[i][1] + 1);
		scores_pair[num_vec - i - 1] = std::make_pair(scores[i], i);
	}

	ind_selected.reserve(num_vec);
	do
	{
		int first = scores_pair.rbegin()->second;
		ind_selected.push_back(first);
		for (typename std::vector<std::pair<T, int> >::iterator it = scores_pair.begin(); it != scores_pair.end();)
		{
			int it_idx = it->second;
			T x1 = std::max(boxes[first][0], boxes[it_idx][0]);
			T y1 = std::max(boxes[first][1], boxes[it_idx][1]);
			T x2 = std::min(boxes[first][2], boxes[it_idx][2]);
			T y2 = std::min(boxes[first][3], boxes[it_idx][3]);

			T w = std::max(T(0.0), x2 - x1 + 1);
			T h = std::max(T(0.0), y2 - y1 + 1);
			T ov = w*h / (area_boxes[first] + area_boxes[it_idx] - w*h);
			if (ov > overlap_thresh)
			{
				it = scores_pair.erase(it);
			}
			else
				it++;
		}
	} while (scores_pair.size() != 0);
}


#endif /* BBOXPROC_HPP */


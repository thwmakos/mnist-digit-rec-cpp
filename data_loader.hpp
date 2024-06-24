//
// ~thwmakos~
//
// Sun 23 Jun 18:35:21 BST 2024
//
// data_loader.hpp
//

#ifndef DATA_LOADER_HPP_INCLUDED
#define DATA_LOADER_HPP_INCLUDED

// load training and test data from `/data` directory
// the data format from https://github.com/sunsided/mnist
// is as follows:
//
// file train-labels-idx1-ubyte:
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
// 0004     32 bit integer  60000            number of items
// 0008     unsigned byte   ??               label
// 0009     unsigned byte   ??               label
// ........
// xxxx     unsigned byte   ??               label
// The labels values are 0 to 9.

// file train-images-idx3-ubyte:
// [offset] [type]          [value]          [description]
// 0000     32 bit integer  0x00000803(2051) magic number
// 0004     32 bit integer  60000            number of images
// 0008     32 bit integer  28               number of rows
// 0012     32 bit integer  28               number of columns
// 0016     unsigned byte   ??               pixel
// 0017     unsigned byte   ??               pixel
// ........
// xxxx     unsigned byte   ??               pixel
// Pixels are organized row-wise. Pixel values are 0 to 255.
// 0 means background (white), 255 means foreground (black).

#include <vector>
#include <string>
#include <cstdint>

namespace thwmakos {

class data_loader
{
    public:
		data_loader(const std::string& image_filename, const std::string& label_filename);
		
		data_loader() : m_num_images(0), m_num_labels(0) {}
		data_loader(const data_loader&) = delete;
		data_loader(data_loader&&) = delete;
		data_loader& operator=(const data_loader&) = delete;
		data_loader& operator=(data_loader&&) = delete;

		~data_loader() = default;

	//private:
		std::vector<std::uint8_t> m_image_data;
		std::vector<int> m_label_data;	
		
		// number of images and labels
		// should always be the same
		int m_num_images;
		int m_num_labels;

		// number of rows and columns per image
		static constexpr int s_num_rows = 28;
		static constexpr int s_num_cols = 28;
		// file of an image, in bytes
		static constexpr int s_image_size = s_num_rows * s_num_cols;
	
		// first 32 bits of training images file
		static constexpr std::int32_t s_image_magic_number = 0x00000803;
		// first 32 bits of label file
		static constexpr std::int32_t s_label = 0x00000801;
		// where does image data start in training image files?
		static constexpr int s_image_file_offset = 16; 
		// where does the label data start in label files?
		static constexpr int s_label_file_offset = 8;
};

} // namespace thwmakos

#endif // DATA_LOADER_HPP_INCLUDED

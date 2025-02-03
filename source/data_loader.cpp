//
// ~thwmakos~
//
// Sun 23 Jun 18:34:34 BST 2024
//
// data_loader.cpp
//
//

#include "data_loader.hpp"
#include "matrix.hpp"

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <format>
//#include <iostream>

namespace thwmakos {

data_loader::data_loader(std::string_view image_filename, std::string_view label_filename)
{
	std::ifstream image_file, label_file;

	// return an ifstream and find out filesize
	auto open_file_and_find_filesize = [] (std::ifstream &file, std::string_view filename)
	{
		std::filesystem::path filepath { filename };
		const auto filesize = std::filesystem::file_size(filepath);
		file.open(filename.data(), std::ios::in | std::ios::binary);
		
		if(!file.is_open())
		{
			throw std::runtime_error(std::format("cannot open file {}", filename));
		}
		
		// seek in the beginning of file to be ready to use
		file.seekg(0, std::ios::beg);
		
		return filesize;
	};

	// read 4 bytes from file and convert them to an int32_t
	auto read_int32_t = [] (std::ifstream& file)
	{
		// the number will be read here
		int32_t number = 0;		
		// 4 bytes of data to be read for file
		std::array<char, 4> number_bytes; 
		
		file.read(number_bytes.data(), number_bytes.size());	

		// the int we read has the most significant byte first 
		// hence we need to reverse the bytes we just read  
		std::reverse(number_bytes.begin(), number_bytes.end());
		// convert bytes to int32_t
		number = *reinterpret_cast<int32_t *>(number_bytes.data());

		return number;
	};
	
	// load training image data
	// convert to int intentionally
	int image_length = open_file_and_find_filesize(image_file, image_filename);

	if(image_length <= s_image_file_offset)
	{
		throw std::runtime_error("empty of malformed file");
	}
	
	// check file header
	// see data_loader.hpp for file structure
	int32_t magic_number = read_int32_t(image_file);

	if(magic_number != s_image_magic_number)
	{
		throw std::runtime_error(std::format("mismatched magic number for file {}", image_filename));
	}

	m_num_images = static_cast<int>(read_int32_t(image_file));
	const auto num_rows = static_cast<int>(read_int32_t(image_file));
	const auto num_cols = static_cast<int>(read_int32_t(image_file));	

	if(s_num_rows != num_rows || s_num_cols != num_cols)
	{
		throw std::runtime_error("expected 28x28 training images");
	}

	// check if file is properly sized
	const auto expected_data_size = m_num_images * s_image_size;
	if(expected_data_size != image_length - s_image_file_offset)
	{
		throw std::runtime_error(std::format("expected data size: {}, actual size: {}",
					expected_data_size, image_length - s_image_file_offset));
	}
		
	// resize vector to fit all the image data
	// there are offset-many bytes in the beginning that are not image data	
	m_image_data.resize(expected_data_size);
	// read form current position to the end of file
	image_file.read(reinterpret_cast<char *>(m_image_data.data()), expected_data_size);

	// load label data
	int label_length = open_file_and_find_filesize(label_file, label_filename);
	// check label magic number
	magic_number = read_int32_t(label_file);	
	
	if(magic_number != s_label_magic_number)
	{
		throw std::runtime_error(std::format("mismatched magic number for file {}", image_filename));
	}
		
	m_num_labels = static_cast<int>(read_int32_t(label_file));

	if(m_num_images != m_num_labels)
	{
		throw std::runtime_error("different number of images and labels");
	}

	const auto expected_label_size = m_num_labels;
	if(expected_label_size != label_length - s_label_file_offset)
	{
		throw std::runtime_error(std::format("expected label size: {}, actual size: {}",
					expected_label_size, label_length - s_label_file_offset));
	}

	m_label_data.resize(expected_label_size);
	label_file.read(reinterpret_cast<char *>(m_label_data.data()), expected_label_size);
}

training_sample data_loader::get_sample(int index) const
{
	if(index < 0 || index >= m_num_images)
	{
		throw std::out_of_range("get_training_sample: index out of range");
	}

	// locate the data in the m_image_data vector
	// each image requires s_image_size elements of a vector	
	auto data_begin = m_image_data.cbegin() + index * s_image_size;
	auto data_end   = data_begin + s_image_size;
	// extract image data
	// using vector here instead of array because we will move 
	// the vector into the return value for usage by caller
	std::vector<FloatType> image_data(data_begin, data_end);
	// each entry in image_data is between 0.0f and 255.0f, which 0 being black and 255 white
	// we need to map these [0.0f, 1.0f]
	std::for_each(image_data.begin(), image_data.end(), [](FloatType &x) { x = x / 255.0f; });
	
	// create the to be returned sample
	// the label and label_val members are set afterwards
	// FIXME: there should not be any unnecessary copies or movies using the training_sample
	// struct
	training_sample sample = { matrix(s_image_size, 1, std::move(image_data)), 
		matrix(10, 1), -1 };
	// set the corresponding entry of label matrix (i.e. column vector) to one
	sample.label_val = static_cast<int>(m_label_data.at(index));
	sample.label[sample.label_val, 0] = 1.0f;

	return sample;
}

} // namespace thwmakos

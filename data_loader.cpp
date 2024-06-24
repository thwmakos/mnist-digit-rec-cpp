//
// ~thwmakos~
//
// Sun 23 Jun 18:34:34 BST 2024
//
// data_loader.cpp
//
//

#include "data_loader.hpp"

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <array>
//#include <iostream>

namespace thwmakos {

data_loader::data_loader(const std::string& image_filename, const std::string& label_filename)
{
	// find out filesize
	std::filesystem::path image_filepath { image_filename };
	// filesize, in bytes
	const auto length = std::filesystem::file_size(image_filepath);

	if(length <= s_image_file_offset)
	{
		throw std::runtime_error("empty of malformed file");
	}

	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);

	if(!image_file.is_open())
	{
		throw std::runtime_error("cannot open training image file");
	}
	
	image_file.seekg(0, std::ios::beg);
	
	// read 4 bytes from file and convert them to an int32_t
	auto read_int32_t = [] (std::ifstream &file)
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

	int32_t magic_number = read_int32_t(image_file);

	if(magic_number != s_image_magic_number)
	{
		throw std::runtime_error("mismatched magic numbers");
	}

	m_num_images = static_cast<int>(read_int32_t(image_file));
	
	if(s_num_rows != static_cast<int>(read_int32_t(image_file)) ||
	  s_num_cols != static_cast<int>(read_int32_t(image_file)))
	{
		throw std::runtime_error("expected 28x28 training images");
	}
		
	// resize vector to fit all the image data
	// there are offset-many bytes in the beginning that are not image data	
	m_image_data.resize(length - s_image_file_offset);
}

} // namespace thwmakos

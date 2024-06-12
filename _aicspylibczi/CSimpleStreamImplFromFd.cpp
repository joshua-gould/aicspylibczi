//
// Created by Jamie Sherman on 11/21/19.
//

#include "CSimpleStreamImplFromFd.h"
#include "exceptions.h"
#include <fcntl.h>
#include <thread>

#ifdef _WIN32
#include <io.h>
#include <stdio.h>
#else
#include <unistd.h>
#endif

namespace pb_helpers {

CSimpleStreamImplFromFd::CSimpleStreamImplFromFd(int file_descriptor_)
  : libCZI::IStream()
{
#ifdef _WIN32
  int dupDesc = _dup(file_descriptor_);
  if (dupDesc == -1) {
    throw pylibczi::FilePtrException("Reader class could not dup the file descriptor!");
  }
  m_fp = _fdopen(dupDesc, "r");
#else
  int dupDesc = dup(file_descriptor_);
  m_fp = fdopen(dupDesc, "r");
#endif
  if (m_fp == nullptr) {
    throw pylibczi::FilePtrException("Reader class received a bad FILE *!");
  }
}

void
CSimpleStreamImplFromFd::Read(std::uint64_t offset_,
                              void* data_ptr_,
                              std::uint64_t size_,
                              std::uint64_t* bytes_read_ptr_)
{
  std::unique_lock<std::mutex> lck(m_mutex);
#ifdef _WIN32
  int r = _fseeki64(this->m_fp, offset_, SEEK_SET);
#else
  int r = fseeko(this->m_fp, offset_, SEEK_SET);
#endif
  if (r != 0) {
    const auto err = errno;
    ostringstream ss;
    ss << "Seek to file-position " << offset_ << " failed, errno=<<" << err << ".";
    throw std::runtime_error(ss.str());
  }
  std::uint64_t bytesRead = fread(data_ptr_, 1, (size_t)size_, this->m_fp);
  if (bytes_read_ptr_ != nullptr)
    (*bytes_read_ptr_) = bytesRead;
}

}

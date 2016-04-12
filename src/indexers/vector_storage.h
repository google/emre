// Copyright 2012-2016 Google
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// An interface (pure virtual classes) for read-only and write-only vectors
// that only support sequential access. This allows us to switch between
// in-memory and disk-based vectors and make the program more scalable on
// a single machine.

#ifndef EMRE_INDEXERS_VECTOR_STORAGE_H_  // NOLINT
#define EMRE_INDEXERS_VECTOR_STORAGE_H_

#include <memory>
#include <vector>

namespace emre {

// Example to sequentially iterate elements in VectorReader:
// VectorReader<string> vec_reader;
// auto itr =  vec_reader.GetIterator();
// while (!itr.Done()) {
//   Process(itr.Next());
// }
// ...
// itr.Reset();
// for (int i = 0; i < vec_reader.Size(); ++i) {
//   OtherProcess(itr.Next());
// }
template <typename T>
class VectorReader {
 public:
  class Iterator {
   public:
    explicit Iterator(VectorReader<T>* vector_reader)
        : vector_reader_(vector_reader),
          cur_idx_(0),
          size_(vector_reader_->Size()),
          vec_(vector_reader_->AsVector()) {
      if (vec_ != nullptr) {
        vec_itr_ = vec_->begin();
      }
      vector_reader_->Reset();
    }

    // Dummy Iterator, Done() will always return true.
    Iterator() {}

    // After this call, Next() will return the first element, i.e., you can
    // iterate from start again.
    void Reset() {
      vector_reader_->Reset();
      cur_idx_ = 0;
      if (vec_ != nullptr) vec_itr_ = vec_->begin();
    }

    // Whether no more elements.
    bool Done() const {
      return cur_idx_ >= size_;
    }

    // Next element.
    // Note: Before call this method, make sure there are still elements to
    // iterate, either check !Done(), or num of iterated elements is smaller
    // than VectorReader.Size(). See VectorReader class comment about example.
    T Next() {
      cur_idx_++;
      // Use vec_itr_ to get next element when possible, which is a little
      // faster than call vector_reader_->Next(), as the later is virtual
      // function. This saves some time for in-memory VectorReader subclasses
      // when Next() is called in a very large loop.
      return vec_ ? *vec_itr_++ : vector_reader_->Next();
    }

   private:
    // The underlying VectorReader. Not owned.
    VectorReader<T>* vector_reader_;
    // Current element index in vector_reader_
    int cur_idx_;
    // vector_reader_ size
    int size_;
    // The vector underlying vector_reader_, which can be used to speed up
    // iteration. Not available to all VectorReader subclasses.
    const vector<T>* vec_;
    // Iterator of vec_. Also see comments in Next().
    typename vector<T>::const_iterator vec_itr_;
  };

  virtual ~VectorReader() {}

  virtual int Size() const = 0;

  // Get a forward iterator to iterate elements.
  // Note: This call invalidats all previous iterators.
  Iterator GetIterator() {
    return Iterator(this);
  }

  // For testing only.
  vector<T> AsVectorForTesting() {
    vector<T> v;
    for (auto itr = this->GetIterator(); !itr.Done();) {
      v.push_back(itr.Next());
    }
    return v;
  }

 protected:
  // The following virtual methods are only used by friend class Iterator, not
  // for public use.

  // Override this method when underlying storage can be cheaply and easily to
  // convert to vector. See comments in Iterator::Next() about why this is
  // needed.
  virtual const vector<T>* AsVector() const { return nullptr; }
  virtual void Reset() = 0;
  virtual T Next() = 0;  // returns the current value and increments the state

 private:
  friend class Iterator;
};

// A VectorReader subclass that contains same value.
template <typename T>
class ConstVectorReader : public VectorReader<T> {
 public:
  explicit ConstVectorReader(T val, int size) : val_(val), size_(size) {}
  virtual ~ConstVectorReader() {}

  int Size() const override { return size_; }

 protected:
  const vector<T>* AsVector() const override { return nullptr; }
  void Reset() override {}
  T Next() override { return val_; }

 private:
  T val_;
  int size_;
};

template <typename T>
class VectorBuilder {
 public:
  virtual ~VectorBuilder() {}
  virtual void Write(T value) = 0;
  virtual void Finish() = 0;  // Same a flush for files
  virtual std::unique_ptr<VectorReader<T>> MoveToReader() = 0;
  virtual int Size() const = 0;
};

}  // namespace emre

#endif  // EMRE_INDEXERS_VECTOR_STORAGE_H_  // NOLINT

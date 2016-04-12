// Copyright 2010-2016 Google
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

#ifndef BASE_NUMBERS_H_  // NOLINT
#define BASE_NUMBERS_H_

#include <string>
#include "integral_types.h"  // NOLINT

namespace emre {
// Convert strings to numerical values.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
bool safe_strtof(const char* str, float* value);
bool safe_strtod(const char* str, double* value);
bool safe_strtof(const std::string& str, float* value);
bool safe_strtod(const std::string& str, double* value);
bool safe_strto32(const std::string& str, int32* value);
bool safe_strto64(const std::string& str, int64* value);
}  // namespace emre

#endif  // BASE_NUMBERS_H_  // NOLINT
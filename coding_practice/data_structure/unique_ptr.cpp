/*
unique_ptr
a) 构造时传入托管对象的指针，析构时delete对象
b) 禁用赋值函数
*/

template<typename T>
class unique_ptr {
private:
    T* ptr;
public:
    // Constructor: Takes a raw pointer and takes ownership
    explicit unique_ptr(T* p = nullptr) : ptr(p) {}         //explicit 防止隐试转换

    ~unique_ptr(){
        delete ptr;
    }

    // Move constructor: Transfers ownership from another unique_ptr
    unique_ptr(unique_ptr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;  // Set the other pointer to nullptr to avoid double delete
    }


};
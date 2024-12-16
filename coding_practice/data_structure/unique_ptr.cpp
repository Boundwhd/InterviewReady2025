/*
下面是一个手写的简化版智能指针（SmartPointer）的实现。这个智能指针实现了以下基本功能：
1. 构造函数：初始化指针并管理资源。
2. 析构函数：自动释放资源。
3. 拷贝构造函数：处理拷贝时，避免多次删除资源。
4. 拷贝赋值运算符：支持赋值操作。
5. -> 操作符重载：让智能指针能够像普通指针一样访问成员。
6. * 操作符重载：让智能指针能够解引用。
*/

#include <iostream>
#include <memory>
using namespace std;

template <typename T>
class Unique_ptr {
private:
    T* ptr;     //原始指针

public:
    //构造函数
    explicit Unique_ptr(T* p = nullptr) : ptr(p) {}

    //析构函数，释放内存
    ~Unique_ptr() {
        delete ptr;
        cout << "Destructor called, memory freed." << endl;
    }

    Unique_ptr(const Unique_ptr& other) = delete;

    Unique_ptr& operator=(const Unique_ptr& other) = delete;

    Unique_ptr(Unique_ptr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;  // 让源对象不再管理资源
        cout << "Move Constructor called." << endl;
    }

    Unique_ptr& operator=(Unique_ptr&& other) noexcept {
        if (this != &other) {
            delete ptr;  // 释放当前管理的资源
            ptr = other.ptr;  // 转移所有权
            other.ptr = nullptr;  // 让源对象不再管理资源
            std::cout << "Move Assignment called." << std::endl;
        }
        return *this;
    }

    T& operator*() const {
        return *ptr;
    }

    T* operator->() const {
        return ptr;
    }

    T* get() const {
        return ptr;
    }

    void reset(T* p = nullptr) {
        delete ptr;
        ptr = p;
    }

    T* release() {
        T* temp = ptr;
        ptr = nullptr;
        return temp;
    }
};

class MyClass {
public:
    int value;

    MyClass(int val) : value(val) {}
    void print() const {
        std::cout << "MyClass value: " << value << std::endl;
    }
};


int main() {
    // 创建一个 MyClass 对象并用 MyUniquePtr 管理
    Unique_ptr<MyClass> ptr1(new MyClass(10));

     // 使用 `*` 和 `->` 操作符
    ptr1->print();
    (*ptr1).print();

    Unique_ptr<MyClass> ptr2 = move(ptr1);
    if (!ptr1.get()) {
        std::cout << "ptr1 is empty after move." << std::endl;
    }

    ptr2.reset(new MyClass(20));
    ptr2->print();
        // 使用 release() 获取原始指针，并让 unique_ptr 不再管理它
    MyClass* rawPtr = ptr2.release();
    rawPtr->print();
    delete rawPtr;
    return 0;
}

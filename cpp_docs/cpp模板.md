# 📌C++ 中的模版（Template）
模板是 C++ 中的泛型编程的基础。 作为强类型语言，C++ 要求所有变量都具有特定类型，由程序员显式声明或编译器推导。 但是，许多数据结构和算法无论在哪种类型上操作，看起来都是相同的。 使用模板可以定义类或函数的操作，并让用户指定这些操作应处理的具体类型。

## 1. 函数模版
允许编写适用于多种数据类型的函数。
```cpp
template <typename T, typename U>
auto multiply(T a, U b) -> decltype(a * b) {    ///decltype自动推断返回类型
    return a * b;
}
```

## 2. 类模版
用于创建具有泛型类型的类
```cpp
template <typename T>
class Box {
private:
    T value;
public:
    void setValue(T val) { value = val; }
    T getValue() { return value; }
};

```

## 3. 模板特化（Template Specialization）
为特定类型提供不同的模板实现。
```cpp
template <typename T>
class Printer {
public:
    void print(T value) { std::cout << value << std::endl; }
};

// 针对 `std::string` 的特化
template <>
class Printer<std::string> {
public:
    void print(std::string value) { std::cout << "String: " << value << std::endl; }
};
```

## 4. 可变参数模板（Variadic Templates）
支持接受不定数量的模板参数。
```cpp
#include <iostream>

template <typename T>
void print(T t) {
    std::cout << t << std::endl;
}

template <typename T, typename... Args>
void print(T t, Args... args) {
    std::cout << t << " ";
    print(args...);
}

int main() {
    print(1, 2, 3, "Hello", 4.5);
    return 0;
}
```

## 5. 模板的默认参数
模板参数可以提供默认值。
```cpp
template <typename T = int>
class DefaultBox {
public:
    T value;
};

DefaultBox<> box;  // 默认使用 int
```


# unique_ptr

std::unique_ptr是C++11起引入的智能指针。为什么必须要在C++11起才有该特性，主要还是C++11增加了move语义，否则无法对对象的所有权进行传递。

### 1. 创建unique_ptr
- unique_ptr不像shared_ptr一样拥有标准库函数make_shared来创建一个shared_ptr实例。要想创建一个unique_ptr，我们需要将一个new操作符返回的指针传递给unique_ptr的构造函数。
- std::make_unique 是 C++14 才有的特性。
```cpp
int main()
{
    // 创建一个unique_ptr实例
    unique_ptr<int> pInt(new int(5));
    cout << *pInt;
}
```

### 3. 无法进行复制构造和赋值操作
- unique_ptr 没有copy构造函数，不支持普通的拷贝和赋值操作。
```cpp
int main() 
{
    // 创建一个unique_ptr实例
    unique_ptr<int> pInt(new int(5));
    unique_ptr<int> pInt2(pInt);    // 报错
    unique_ptr<int> pInt3 = pInt;   // 报错
}
```

### 4. 可以进行移动构造和移动赋值操作
- unique_ptr虽然没有支持普通的拷贝和赋值操作，但却提供了一种移动机制来将指针的所有权从一个unique_ptr转移给另一个unique_ptr.如果需要转移所有权，可以使用std::move()函数。
```cpp
int main() 
{
    unique_ptr<int> pInt(new int(5));
    unique_ptr<int> pInt2 = std::move(pInt);    // 转移所有权
    //cout << *pInt << endl; // 出错，pInt为空
    cout << *pInt2 << endl;
    unique_ptr<int> pInt3(std::move(pInt2));
}
```

### 5. 可以返回unique_ptr
- unique_ptr不支持拷贝操作，但却有一个例外：可以从函数中返回一个unique_ptr。

```cpp
unique_ptr<int> clone(int p)
{
    unique_ptr<int> pInt(new int(p));
    return pInt;    // 返回unique_ptr
}

int main() {
    int p = 5;
    unique_ptr<int> ret = clone(p);
    cout << *ret << endl;
}
```

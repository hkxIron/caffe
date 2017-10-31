// shared_ptr::reset example
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <memory>
#include <vector>

int main() {
  boost::shared_ptr<int> sp;  // empty

  sp.reset(new int);  // takes ownership of pointer
  *sp = 10;
  std::cout << *sp << '\n';

  sp.reset(new int);  // deletes managed object, acquires new pointer
  //*sp = 20;
  std::cout << *sp << '\n';  // 看来reset时，并不会将原来的值复制过去

  sp.reset();  // deletes managed object

  std::vector<int> a(2, 3);
  std::vector<int> b(2, 4);
  std::cout << "a equals b:" << (a == b) << std::endl;
  for (std::vector<int>::iterator it = a.begin(); it != a.end(); ++it)
    std::cout << ' ' << *it;
  std::cout << '\n';

  return 0;
}

/*删除排序链表中的重复元素II*/

#include <iostream>
#include <stdio.h>

using namespace std;

struct ListNode{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

void printList(ListNode *head){
    while(head){
        cout << head->val << "->";
        head = head->next;
    }
    cout << "NULL" << endl;
}

ListNode* deleteDuplicates(ListNode* head){
    ListNode* dummy = new ListNode(0);
    dummy->next = head;
    ListNode* p = dummy;
    while(p->next){
        ListNode* q = p->next;
        while(q->next && q->next->val == q->val){
            q = q->next;
        }
        if(p->next == q){
            p = p->next;
        }else{
            p->next = q->next;
        }
    }
    return dummy->next;
}

int main() {
    ListNode* l1 = new ListNode(1);
    l1->next = new ListNode(2);
    l1->next->next = new ListNode(2);
    l1->next->next->next = new ListNode(3);
    l1->next->next->next->next = new ListNode(3);
    l1->next->next->next->next->next = new ListNode(4);
    l1->next->next->next->next->next->next = new ListNode(6);

    ListNode* ans = deleteDuplicates(l1);
    printList(ans);
    return 0;
}



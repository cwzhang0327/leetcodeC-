// Source : https://oj.leetcode.com/problems/add-two-numbers/
// Author : Hao Chen
// Date   : 2014-06-18

/********************************************************************************** 
* 
* You are given two linked lists representing two non-negative numbers. 
* The digits are stored in reverse order and each of their nodes contain a single digit. 
* Add the two numbers and return it as a linked list.
* 
* Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
* Output: 7 -> 0 -> 8
*               
**********************************************************************************/

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        //variable initialization
        ListNode* tempNode = new ListNode;
        ListNode* tail = tempNode;
        int digit1, digit2, carry = 0;

        //loop to add numbers and create new list
        while(l1 != nullptr or l2 != nullptr or carry != 0){ //stops when at end of both lists and no carry
            if(l1 != nullptr){
                digit1 = l1 -> val;
            }
            else{
                digit1 = 0;
            }

            if(l2 != nullptr){
                digit2 = l2 -> val;
            }
            else{
                digit2 = 0;
            }

            // sum = digit in both linked list plus if there was a carry
            int sum = digit1 + digit2 + carry;
            // digit in new linked list, will cap at 9
            int digit = sum % 10;
            // carry over a 1 if sum is greater than 10
            if(sum >= 10){
                carry = 1;
            }
            else{
                carry = 0;
            }

            tail -> next = new ListNode(digit);
            tail = tail -> next;

            if (l1 != nullptr) {
                l1 = l1 -> next;
            }
            if (l2 != nullptr){
                l2 = l2 -> next;
            }
        }
        ListNode* result = tempNode -> next;
        //free memory from temp node
        delete tempNode;
        return result;
    }
};

#! /usr/bin/env python
# -*- coding: utf-8 -*-
def get_new_label_dict():
    str_list = u"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ,/:abcdefghijklmnopqrstuvwxyz";
    new_label_dict = {}
    c = 0
    for i in str_list:
        new_label_dict[c] = i
        c = c + 1
    print('label size:' + str(len(str_list)))
    return new_label_dict

label_dict = get_new_label_dict()
small_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ,/:abcdefghijklmnopqrstuvwxyz'  # type: str

package com.gmimg.multicampus.springboot.member;

import lombok.Data;

@Data
public class Item {
    private int iditem;
    private int memIdx;
    private String filename;
    private float acc;
    private int del;
}
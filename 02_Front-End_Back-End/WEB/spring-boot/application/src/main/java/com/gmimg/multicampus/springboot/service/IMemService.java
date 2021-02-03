package com.gmimg.multicampus.springboot.service;

import com.gmimg.multicampus.springboot.member.Member;

public interface IMemService {

	public void insertMem(Member member) throws Exception;
	public Member findMem(Member member) throws Exception;
	public Member idCheckMem(Member member) throws Exception;
}

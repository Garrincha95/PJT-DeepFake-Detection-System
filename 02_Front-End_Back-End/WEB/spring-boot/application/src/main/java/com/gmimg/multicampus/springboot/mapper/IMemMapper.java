package com.gmimg.multicampus.springboot.mapper;

import org.apache.ibatis.annotations.Mapper;

import com.gmimg.multicampus.springboot.member.Member;

@Mapper
public interface IMemMapper {
	
	public void insertMem(Member member) throws Exception;
	public Member findMem(Member member) throws Exception;
	public Member idCheckMem(Member member) throws Exception;

}

package com.gmimg.multicampus.springboot.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.gmimg.multicampus.springboot.mapper.IMemMapper;
import com.gmimg.multicampus.springboot.member.Member;

import lombok.AllArgsConstructor;

@Service
public class MemService implements IMemService {

	@Autowired
	IMemMapper mapper;
	
	@Override
	public void insertMem(Member member) throws Exception {
		mapper.insertMem(member);
	}

	@Override
	public Member findMem(Member member) throws Exception {
		// TODO Auto-generated method stub
		return mapper.findMem(member);
	}

	@Override
	public Member idCheckMem(Member member) throws Exception {
		// TODO Auto-generated method stub
		return mapper.idCheckMem(member);
	}

}

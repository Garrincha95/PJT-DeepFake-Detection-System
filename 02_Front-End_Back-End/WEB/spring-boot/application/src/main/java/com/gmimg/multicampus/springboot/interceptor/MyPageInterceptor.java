package com.gmimg.multicampus.springboot.interceptor;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import org.springframework.stereotype.Component;
import org.springframework.web.servlet.handler.HandlerInterceptorAdapter;

import com.gmimg.multicampus.springboot.member.Member;

@Component
public class MyPageInterceptor extends HandlerInterceptorAdapter{

	@Override
	public boolean preHandle(HttpServletRequest request,
			HttpServletResponse response, Object object) throws Exception{
		
		HttpSession session = request.getSession();
		
		Member member  = (Member) session.getAttribute("sessionMem");
		
		if (member == null) {
			response.sendRedirect("/member/loginForm");
			return false;
		}
		
		return true;
	}
}

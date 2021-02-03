package com.gmimg.multicampus.springboot;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.support.SessionStatus;
import org.springframework.web.servlet.ModelAndView;

import com.gmimg.multicampus.springboot.member.Member;
import com.gmimg.multicampus.springboot.service.MemService;

@Controller
public class MemberController {

	@Autowired
	MemService service; 
	
	//spring security 내의  비밀번호 인코딩 기능 
	@Autowired
	PasswordEncoder passwordEncoder;
	
	//회원가입 페이지 매핑
	@RequestMapping(value = "/member/registerForm", method = RequestMethod.GET)
	public String registerForm() {
		//templates내의 register.html로 연결
		return "register";
	}
	// 회원가입 폼 제출시 DB와 연결시켜주는 매핑 
	@RequestMapping(value = "/register", method = RequestMethod.POST)
	public ModelAndView register(ModelAndView mav, Member member) throws Exception {
		
		//사용자가 입력한 password 값
		String inputPass = member.getMemPw();
		//사용자의 password 인코딩
		String pass = passwordEncoder.encode(inputPass);
		//사용자의 password 인코딩된 값으로 대체
		member.setMemPw(pass);
		//사용자가 입력한 이메일과 인코딩 된 password를 DB에 저장 		
		service.insertMem(member);
		//모든 작업 수행후 메인 페이지로 리다이렉트
		mav.setViewName("redirect:/");
		
		return mav;
	}
	
	//회원가입시 아이디 중복확인 
	@ResponseBody
	@RequestMapping(value = "/idChk", method = {RequestMethod.POST,RequestMethod.GET})
	public int postIdCheck(Member member) throws Exception {
		//사용자가 입력한 아이디가 DB에 등록되어있는지 확인
		Member check = service.idCheckMem(member);
		//result의 기본값을 0으로 설정
		int result = 0;
		if (check != null) {
			//사용자가 입력한 아이디가 DB에 등록되어 있지 않다면 result의 값이 1로 변화
			result = 1;
		} //사용자가 입력한 아이디가 DB에 등록되어 있다면 result의 값은 그대로 0
		//사용자가 입력한 아이디가 DB에 등록되어 있는 여부에 따라서 result값 전송
		return result;
	}

	//로그인 페이지 매핑
	@RequestMapping(value = "/member/loginForm")
	public String loginForm() {
		//templates내의 login.html로 연결
		return "login";
	}
	
	// 로그인 폼 제출시 DB에 확인후 로그인 시켜주는 매핑  
	@RequestMapping(value = "/login", method = RequestMethod.POST)
	public ModelAndView login(ModelAndView mav, Model model,
			Member member, HttpServletRequest request, HttpServletResponse response) throws Exception {
		HttpSession session = request.getSession();
		//사용자가 입력한 이메일이 DB에 등록되어있는지 확인
		Member mem = service.findMem(member);
		
		if (mem == null) {
			//사용자가 입력한 이메일이 DB에 등록되어 있지 않다면 로그인 페이지로 리다이렉트 
			mav.setViewName("login");
		} else {
			//사용자가 입력한 이메일이 DB에 등록되어 있다면 
			//그 등록된 아이디의 인코딩된 password와 입력한 password를 비교하여 T/F로 출력
			boolean passMatch = passwordEncoder.matches(member.getMemPw(), mem.getMemPw());
			if (passMatch) {
				//T 이면 sessionMem 세션객체에 사용자가 입력한 정보를 적용해줌 
				session.setAttribute("sessionMem", mem);
				//메인페이지로 리다이렉트
				mav.setViewName("index");			
			} else {
				//F 이면 sessionMem 세션객체에 null값이 들어가게됨
				session.setAttribute("sessionMem", null);
				//로그인이 실패했으므로 로그인 페이지로 리다이렉트
				mav.setViewName("login");
			}
		}
		return mav;
	}
	//로그아웃 버튼 매핑
	@RequestMapping(value = "/member/logOut")
	public String logOut(HttpServletRequest request, SessionStatus status) {
		
		HttpSession session = request.getSession();
		//현재 페이지에 적용되어있는 session정보를 모두삭제 이 app에서는 session을 하나만 사용하기 때문에 지정하지 않고 모두삭제함
		session.invalidate();
		//이후 메인페이지로 리다이렉트
		return "redirect:/";
	}
	
	
}

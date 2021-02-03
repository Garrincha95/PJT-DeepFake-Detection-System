package com.gmimg.multicampus.springboot.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.WebSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

import lombok.AllArgsConstructor;

@Configuration
@EnableWebSecurity
@AllArgsConstructor
public class SecurityConfig extends WebSecurityConfigurerAdapter{

	@Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
	
	@Override
	public void configure(WebSecurity web) throws Exception {
        web.ignoring()
                .antMatchers("/**");
    }
//	@Override
//	protected void configure(HttpSecurity http) throws Exception {
//	    http.authorizeRequests()
//	    		.antMatchers("/**").permitAll()
//	    	.and()
//	        	.logout()
//	        	.logoutUrl("/member/logOut")
//	        	.logoutSuccessUrl("/")
//	        	.invalidateHttpSession(true)
//	        	.permitAll();
//	        .and()
//	        	.formLogin()
//	        	.loginPage("/member/loginForm")
//	        	.defaultSuccessUrl("/")
//	        	.usernameParameter()
//	        	.permitAll();
//	
//	}



}

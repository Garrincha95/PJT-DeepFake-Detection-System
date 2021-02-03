package com.gmimg.multicampus.springboot.mapper;

import java.util.List;

import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import com.gmimg.multicampus.springboot.member.Item;

@Mapper
public interface IItemMapper {

	// @Insert("insert into item (memIdx, filename) values (#{memIdx}, #{filename})")
	// Item insertItem(@Param("memId") int memIdx, @Param("filename") String filename);

	@Select("select * from item where memIdx = #{memIdx}")
	List<Item> findItem(@Param("memIdx") int memIdx);

	@Update("update item set del=1 where iditem = #{iditem}")
	void deleteItem(@Param("iditem") String iditem);

	@Update("update item set score=#{score} where iditem = #{f} and memIdx = #{id}")
	void star(@Param("f") String f, @Param("score") String score, @Param("id") Integer id);

}

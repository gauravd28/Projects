package org.ChatApplication.data.entity;

/**
 * 
 * @author Komal
 *
 */
public class UserVO {

	private int id;

	private String ninerId;

	private String firstName;

	private String lastName;

	private String email;

	private String contact;

	private String password;

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public String getFirstName() {
		return firstName;
	}

	public void setFirstName(String firstName) {
		this.firstName = firstName;
	}

	public String getLastName() {
		return lastName;
	}

	public void setLastName(String lastName) {
		this.lastName = lastName;
	}

	public String getEmail() {
		return email;
	}

	public void setEmail(String email) {
		this.email = email;
	}

	public String getPassword() {
		return password;
	}

	public void setPassword(String password) {
		this.password = password;
	}

	public String getNinerId() {
		return ninerId;
	}

	public void setNinerId(String ninerId) {
		this.ninerId = ninerId;
	}

}

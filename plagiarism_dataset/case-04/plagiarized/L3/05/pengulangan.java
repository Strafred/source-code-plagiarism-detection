import java.util.*;
class inputoutput{
	public static void main(String[] args) 
	{
		int m = 1;
	System.out.println("Miles\t\tKilometers");
	System.out.println("-------------------------------");
		while (m <= 10) {
		System.out.println(m + "\t\t" + m * 1.609);
		m++;
		}
	}
}
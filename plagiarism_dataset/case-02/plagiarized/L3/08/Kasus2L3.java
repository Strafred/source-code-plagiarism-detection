import java.util.Scanner;
public class Kasus2L3 {
    public static void main(String[] args) {
        System.out.print("Enter the radius and length of a cylinder: ");
        Scanner in = new Scanner(System.in);
        double radius = in.nextDouble();
        double length = in.nextDouble();
        double luas = radius * radius * 3.14159;
        System.out.println("The area is " + luas);
        double volume = luas * length;
        System.out.println("The volume of the cylinder is " + volume);
    }
}

import java.util.Scanner;
public class Kasus2L2 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder: ");
        float radius = in.nextFloat();
        float length = in.nextFloat();
        double luas = radius * radius * 3.14159;
        double volume = luas * length;
        System.out.println("The area is " + luas);
        System.out.println("The volume of the cylinder is " + volume);
    }
}

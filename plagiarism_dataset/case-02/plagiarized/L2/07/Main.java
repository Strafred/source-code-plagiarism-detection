import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder: ");
        double jejari = input.nextDouble();
        double length = input.nextDouble();
        double area = jejari * jejari * 3.14159;
        double volume = area * length;
        System.out.println("The area is " + area);
        System.out.println("The volume of the cylinder is " + volume);
        }
}
